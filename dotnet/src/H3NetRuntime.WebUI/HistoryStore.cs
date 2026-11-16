using System.Text.Json;
using System.Text.Json.Serialization;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Append-only generation history persisted next to the outputs. It only
/// stores summaries (no prompts beyond a truncated preview, no raw JSON), so
/// the file stays small enough to rewrite atomically on every finished job.
/// </summary>
public sealed class HistoryStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly string _path;
    private readonly int _max;
    private readonly ILogger<HistoryStore> _logger;
    private readonly object _sync = new();
    private readonly List<JobSummary> _items = new();

    public HistoryStore(WebUiOptions options, ILogger<HistoryStore> logger)
    {
        _path = options.HistoryFile;
        _max = Math.Max(1, options.MaxHistory);
        _logger = logger;
        Load();
    }

    public IReadOnlyList<JobSummary> Snapshot()
    {
        lock (_sync) return _items.ToArray();
    }

    public void Add(JobSummary item)
    {
        lock (_sync)
        {
            _items.RemoveAll(x => x.Id == item.Id);
            _items.Insert(0, item);
            if (_items.Count > _max) _items.RemoveRange(_max, _items.Count - _max);
            Save();
        }
    }

    public bool Remove(string id)
    {
        lock (_sync)
        {
            int removed = _items.RemoveAll(x => x.Id == id);
            if (removed == 0) return false;
            Save();
            return true;
        }
    }

    private void Load()
    {
        try
        {
            if (!File.Exists(_path)) return;
            string json = File.ReadAllText(_path);
            if (string.IsNullOrWhiteSpace(json)) return;
            List<JobSummary>? items = JsonSerializer.Deserialize<List<JobSummary>>(json, JsonOptions);
            if (items is null) return;
            _items.AddRange(items.Where(x => !string.IsNullOrEmpty(x.Id)).Take(_max));
            _logger.LogInformation("history: {Count} entries from {Path}", _items.Count, _path);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "cannot read history file {Path}", _path);
        }
    }

    private void Save()
    {
        try
        {
            string json = JsonSerializer.Serialize(_items, JsonOptions);
            string temp = _path + ".tmp";
            File.WriteAllText(temp, json);
            File.Move(temp, _path, overwrite: true);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "cannot write history file {Path}", _path);
        }
    }
}
