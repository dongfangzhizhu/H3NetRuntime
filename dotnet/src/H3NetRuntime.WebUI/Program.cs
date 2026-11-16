using H3NetRuntime;
using H3NetRuntime.WebUI;
using Microsoft.Extensions.FileProviders;

var switchMappings = new Dictionary<string, string>
{
    ["--edition"] = "H3:Edition",
    ["--port"] = "H3:Port",
    ["--host"] = "H3:Host",
    ["--model-dir"] = "H3:ModelDir",
    ["--model-root"] = "H3:ModelRoot",
    ["--config"] = "H3:ConfigPath",
    ["--output"] = "H3:OutputDir",
    ["--preload"] = "H3:PreloadOnStart"
};

var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    // Resolve appsettings.json and wwwroot next to the binary, so the app can
    // be started from any working directory (shortcut, service, CI step).
    ContentRootPath = AppContext.BaseDirectory,
    WebRootPath = Path.Combine(AppContext.BaseDirectory, "wwwroot")
});
builder.Configuration.AddCommandLine(args, switchMappings);

var options = new WebUiOptions();
builder.Configuration.GetSection("H3").Bind(options);
options.Normalize(Directory.GetCurrentDirectory());

builder.WebHost.UseUrls($"http://{options.Host}:{options.Port}");
builder.WebHost.ConfigureKestrel(kestrel => kestrel.Limits.MaxRequestBodySize = 512L * 1024 * 1024);

builder.Services.AddSingleton(options);
builder.Services.AddSingleton<RuntimeHost>();
builder.Services.AddSingleton<HistoryStore>();
builder.Services.AddSingleton<JobManager>();
builder.Services.ConfigureHttpJsonOptions(json =>
{
    json.SerializerOptions.DefaultIgnoreCondition =
        System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
});

WebApplication app = builder.Build();

var startupLogger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("startup");
startupLogger.LogInformation("H3 WebUI edition={Edition} port={Port} model={Model}",
    options.Edition, options.Port, options.ModelDir ?? "(auto)");

var jobs = app.Services.GetRequiredService<JobManager>();
jobs.Start();

app.UseDefaultFiles();
app.UseStaticFiles();

// Outputs (MP4 + preview PNG) and uploaded references are served from disk so
// the browser can stream them without an extra round trip through the API.
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(options.OutputDir),
    RequestPath = "/files/outputs",
    OnPrepareResponse = ctx => ctx.Context.Response.Headers.CacheControl = "no-cache"
});
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(options.UploadDir),
    RequestPath = "/files/uploads"
});

// ------------------------------------------------------------------- config

app.MapGet("/api/config", (WebUiOptions o, RuntimeHost host) => new ConfigResponse
{
    Edition = o.Edition,
    AdvancedEnabled = o.Advanced,
    Version = H3Runtime.Version,
    AbiVersion = H3Runtime.AbiVersion,
    ModelLoaded = host.IsLoaded,
    ModelError = host.LastError,
    ModelDir = host.ModelDir,
    OutputDir = o.OutputDir.Replace('\\', '/'),
    BuildInfoJson = H3Runtime.BuildInfoJson,
    Presets = Presets.All.ToList(),
    Resolutions = Presets.Resolutions.ToList(),
    FrameChoices = Presets.FrameChoices.ToList(),
    Packs = host.ScanPacks(),
    UploadEnabled = true,
    MaxFrames = 362,
    DefaultPort = o.Port
});

app.MapGet("/api/status", (RuntimeHost host, JobManager queue) =>
{
    bool loaded = host.TryEnsure(out _, out string? error);
    return new StatusResponse
    {
        ModelLoaded = loaded,
        ModelError = error,
        ModelDir = host.ModelDir,
        DeviceInfoJson = host.DeviceInfoJson,
        ModelInfoJson = host.ModelInfoJson,
        CacheInfoJson = host.CacheInfoJson,
        Busy = queue.Active().Any(j => j.Status is "running" or "cancelling")
    };
});

app.MapGet("/api/health", () => Results.Ok(new { status = "ok", at = DateTimeOffset.Now }));

// ------------------------------------------------------------------- jobs

app.MapPost("/api/generate", (GenerateApiRequest request, JobManager queue) =>
{
    try
    {
        JobView job = queue.Submit(request);
        return Results.Json(job, statusCode: StatusCodes.Status202Accepted);
    }
    catch (ArgumentException ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
    catch (InvalidOperationException ex)
    {
        return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status503ServiceUnavailable);
    }
    catch (H3RuntimeException ex)
    {
        return Results.Json(new { error = ex.Message, status = (int)ex.Status },
            statusCode: StatusCodes.Status503ServiceUnavailable);
    }
});

app.MapGet("/api/jobs", (JobManager queue) => queue.Active());

app.MapGet("/api/jobs/{id}", (string id, JobManager queue) =>
    queue.Get(id) is { } job ? Results.Ok(job) : Results.NotFound(new { error = "任务不存在。" }));

app.MapPost("/api/jobs/{id}/cancel", (string id, JobManager queue) =>
    queue.Cancel(id) ? Results.Ok(new { cancelled = true }) : Results.NotFound(new { error = "任务不存在或已结束。" }));

app.MapGet("/api/history", (JobManager queue) => queue.History());

// ----------------------------------------------------------------- engine ops

app.MapPost("/api/preload", (PreloadApiRequest? body, JobManager queue) =>
{
    try
    {
        JobManager.PreloadOutcome outcome = queue.Preload(body ?? new PreloadApiRequest());
        return Results.Ok(new PreloadResponse
        {
            ElapsedMs = Math.Round(outcome.ElapsedMs),
            ModelDir = outcome.ModelDir
        });
    }
    catch (InvalidOperationException ex)
    {
        return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status409Conflict);
    }
    catch (H3RuntimeException ex)
    {
        return Results.Json(new { error = ex.Message, status = (int)ex.Status },
            statusCode: StatusCodes.Status500InternalServerError);
    }
});

app.MapPost("/api/model/select", (ModelSelectRequest body, WebUiOptions o, RuntimeHost host, JobManager queue) =>
{
    if (!o.Advanced)
        return Results.Json(new { error = "通用版不支持切换模型包。" }, statusCode: StatusCodes.Status403Forbidden);
    if (queue.Active().Count > 0)
        return Results.Json(new { error = "请等当前任务结束后再切换模型。" }, statusCode: StatusCodes.Status409Conflict);
    if (string.IsNullOrWhiteSpace(body.ModelDir))
        return Results.BadRequest(new { error = "modelDir 不能为空。" });
    try
    {
        host.Recreate(body.ModelDir, body.ConfigPath);
        return Results.Ok(new { modelDir = host.ModelDir });
    }
    catch (Exception ex)
    {
        return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status400BadRequest);
    }
});

app.MapPost("/api/cache/clear", (RuntimeHost host) =>
{
    try
    {
        host.Ensure().ClearCache();
        return Results.Ok(new { cleared = true, info = host.CacheInfoJson });
    }
    catch (Exception ex)
    {
        return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status500InternalServerError);
    }
});

// ------------------------------------------------------------------- upload

string[] allowedExtensions =
{
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg"
};

app.MapPost("/api/upload", async (HttpRequest request, WebUiOptions o) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "需要 multipart/form-data 请求。" });

    IFormCollection form;
    try { form = await request.ReadFormAsync(); }
    catch (Exception ex) { return Results.BadRequest(new { error = "无法读取上传内容：" + ex.Message }); }

    IFormFile? file = form.Files.Count > 0 ? form.Files[0] : null;
    if (file is null || file.Length == 0)
        return Results.BadRequest(new { error = "没有收到文件。" });

    string extension = Path.GetExtension(file.FileName).ToLowerInvariant();
    if (!allowedExtensions.Contains(extension))
        return Results.BadRequest(new { error = $"不支持的文件类型 {extension}。" });

    string baseName = Path.GetFileNameWithoutExtension(file.FileName);
    var safe = new System.Text.StringBuilder(baseName.Length);
    foreach (char c in baseName)
        safe.Append(char.IsLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_');
    string stored = $"{Guid.NewGuid():N}"[..8] + "-" + safe.ToString().Trim('.', '_') + extension;
    if (stored.Length > 96) stored = stored[..88] + extension;

    string path = Path.Combine(o.UploadDir, stored);
    await using (FileStream stream = File.Create(path))
        await file.CopyToAsync(stream);

    return Results.Ok(new UploadResponse
    {
        Path = path.Replace('\\', '/'),
        Name = file.FileName,
        Size = file.Length,
        Url = "/files/uploads/" + Uri.EscapeDataString(stored)
    });
}).DisableAntiforgery();

// ------------------------------------------------------------------- startup

if (options.PreloadOnStart)
{
    _ = Task.Run(async () =>
    {
        try
        {
            await Task.Delay(500);
            RuntimeHost host = app.Services.GetRequiredService<RuntimeHost>();
            H3Runtime runtime = host.Ensure();
            startupLogger.LogInformation("model pack ready: {Dir}", host.ModelDir);
            _ = runtime;
        }
        catch (Exception ex)
        {
            startupLogger.LogError(ex, "preload on start failed");
        }
    });
}

app.Lifetime.ApplicationStopping.Register(() =>
{
    app.Services.GetRequiredService<JobManager>().Dispose();
    app.Services.GetRequiredService<RuntimeHost>().Dispose();
});

app.Run();
