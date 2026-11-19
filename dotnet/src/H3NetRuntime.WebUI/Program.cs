using System.Text;
using H3NetRuntime;
using H3NetRuntime.WebUI;
using Microsoft.Extensions.FileProviders;

// The demo/executable messages are Chinese. On a Windows console the default
// OEM code page renders them as mojibake (and redirecting to a log file keeps
// the damage); force UTF-8 so console and piped logs stay readable everywhere.
try { Console.OutputEncoding = new UTF8Encoding(false); }
catch (Exception) { /* redirected streams on some hosts refuse a re-encode */ }

var switchMappings = new Dictionary<string, string>
{
    ["--edition"] = "H3:Edition",
    ["--port"] = "H3:Port",
    ["--host"] = "H3:Host",
    ["--model-dir"] = "H3:ModelDir",
    ["--model-root"] = "H3:ModelRoot",
    ["--config"] = "H3:ConfigPath",
    ["--output"] = "H3:OutputDir",
    ["--preload"] = "H3:PreloadOnStart",
    ["--keep-resident-vae"] = "H3:KeepResidentVae",
    ["--keep-dit"] = "H3:KeepDit"
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
// Working dirs follow the launch directory; model paths follow appsettings.
options.Normalize(Directory.GetCurrentDirectory(), AppContext.BaseDirectory);

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

// Print the resolved component paths before the first request: the engine
// loads whatever these point at, and a wrong pack is otherwise only visible as
// a bad video an hour later.
{
    RuntimeHost probe = app.Services.GetRequiredService<RuntimeHost>();
    foreach (string line in probe.Discovery.Describe()) startupLogger.LogInformation("{Line}", line);
}

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

app.MapGet("/api/config", (WebUiOptions o, RuntimeHost host) =>
{
    ModelDiscovery discovery = host.Discovery;
    return new ConfigResponse
    {
        Edition = o.Edition,
        AdvancedEnabled = o.Advanced,
        Version = H3Runtime.Version,
        AbiVersion = H3Runtime.AbiVersion,
        ModelLoaded = host.IsLoaded,
        // The engine has not been touched yet at this point, so the error comes
        // from the file probe: the shell can explain a missing pack on first paint.
        ModelError = host.LastError ?? discovery.Error,
        ModelDir = Slash(discovery.PackDir ?? host.ModelDir),
        ModelPackDir = Slash(discovery.PackDir ?? string.Empty),
        OutputDir = Slash(o.OutputDir),
        BuildInfoJson = H3Runtime.BuildInfoJson,
        Presets = Presets.All.ToList(),
        Resolutions = Presets.Resolutions.ToList(),
        FrameChoices = Presets.FrameChoices.ToList(),
        Packs = host.ScanPacks(),
        UploadEnabled = true,
        MaxFrames = 362,
        DefaultPort = o.Port,
        Modes = GenerationModes.All.ToList(),
        Ref2vaAvailable = discovery.Ref2VaPresent,
        ModelComponents = discovery.Components.Select(c => new ModelComponentView
        {
            Key = c.Key,
            Label = c.Label,
            Slot = c.Slot,
            Path = Slash(c.Path ?? string.Empty),
            Required = c.Required,
            Present = c.Present
        }).ToList(),
        ModelSearchPaths = discovery.Searched.Select(Slash).ToList(),
        VaePath = VaePathOf(o)
    };
});

app.MapGet("/api/status", (WebUiOptions o, RuntimeHost host, JobManager queue) =>
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
        Busy = queue.Active().Any(j => j.Status is "running" or "cancelling"),
        VaePath = VaePathOf(o)
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

/// Re-probes the disk after the user drops weights in place; does not touch a
/// loaded engine, so it is safe to call at any time.
app.MapPost("/api/model/refresh", (RuntimeHost host) =>
{
    host.Refresh();
    ModelDiscovery discovery = host.Discovery;
    return Results.Ok(new
    {
        usable = discovery.IsUsable,
        packDir = Slash(discovery.PackDir ?? string.Empty),
        ref2va = discovery.Ref2VaPresent,
        error = discovery.Error,
        missing = discovery.Missing.Select(c => c.Label).ToList()
    });
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

/// <summary>Windows paths read better with forward slashes in JSON and HTML.</summary>
static string Slash(string path) => path.Replace('\\', '/');

/// <summary>
/// The effective video-VAE path. "resident-slow" means the ~9.5 GiB decoder is
/// kept in VRAM between jobs; on a 24 GiB card that can leave no headroom and
/// stall a consecutive generation. "fast" is the safe one-shot default.
/// </summary>
static string VaePathOf(WebUiOptions o) => o.KeepResidentVae ? "resident-slow" : "fast";
