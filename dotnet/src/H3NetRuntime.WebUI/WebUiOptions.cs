namespace H3NetRuntime.WebUI;

public sealed class WebUiOptions
{
    public string Edition { get; set; } = "general";
    public string OutputDir { get; set; } = "work/outputs";
    public string UploadDir { get; set; } = "work/uploads";
    public string HistoryFile { get; set; } = "work/history.json";
    public string? ModelDir { get; set; }
    public string? ConfigPath { get; set; }
    public string? ModelRoot { get; set; }
    public bool PreloadOnStart { get; set; }
    public int MaxHistory { get; set; } = 60;
    public int Port { get; set; } = 7867;

    /// <summary>Bind address; loopback by default, use 0.0.0.0 to expose on LAN.</summary>
    public string Host { get; set; } = "127.0.0.1";

    public bool Advanced => string.Equals(Edition, "advanced", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Makes every path absolute and creates the working directories. Relative
    /// paths resolve against <paramref name="baseDirectory"/> (the content
    /// root) so the app behaves the same however it was launched.
    /// </summary>
    public void Normalize(string baseDirectory)
    {
        Edition = Advanced ? "advanced" : "general";
        if (string.IsNullOrWhiteSpace(Host)) Host = "127.0.0.1";
        OutputDir = Resolve(baseDirectory, OutputDir);
        UploadDir = Resolve(baseDirectory, UploadDir);
        HistoryFile = Resolve(baseDirectory, HistoryFile);
        if (!string.IsNullOrWhiteSpace(ModelDir)) ModelDir = Resolve(baseDirectory, ModelDir);
        if (!string.IsNullOrWhiteSpace(ConfigPath)) ConfigPath = Resolve(baseDirectory, ConfigPath);
        if (!string.IsNullOrWhiteSpace(ModelRoot)) ModelRoot = Resolve(baseDirectory, ModelRoot);
        Directory.CreateDirectory(OutputDir);
        Directory.CreateDirectory(UploadDir);
        Directory.CreateDirectory(Path.GetDirectoryName(HistoryFile) ?? OutputDir);
    }

    private static string Resolve(string baseDirectory, string path) =>
        Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(baseDirectory, path));
}
