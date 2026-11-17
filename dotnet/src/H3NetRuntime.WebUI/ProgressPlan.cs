using System.Collections.Concurrent;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Turns the engine's per-phase progress into one overall number the page can
/// draw, plus a Chinese label for the console and the status line.
///
/// The engine reports each stage on its own scale ("text encoder 12/50",
/// "denoise 3/8", "video VAE load 20/24"), so a bar fed straight from
/// <c>completed / total</c> restarts from zero five times per run.
///
/// Two properties matter more than precision:
/// <list type="bullet">
/// <item>the stage order differs per mode — a reference job encodes the
/// reference images and the Qwen vision tower <em>before</em> the text encoder,
/// while a text job goes straight to the encoder — so the plan cannot be a
/// running sum of "every stage before this one";</item>
/// <item>a stage may be reported twice (two reference images) and a cached run
/// may skip stages entirely, so the bar must never move backwards.</item>
/// </list>
/// <see cref="Locate"/> therefore accumulates the best fraction reached per
/// stage and weights the sum, which is order-independent and monotonic by
/// construction.
/// </summary>
public static class ProgressPlan
{
    public sealed record Stage(string Key, string Label, double Weight, params string[] Aliases);

    /// <summary>Text and first/last-frame jobs: the encoder runs first.</summary>
    private static readonly Stage[] SharedStages =
    {
        new("prepare", "准备", 0.005),
        new("tokenizer", "分词", 0.005),
        new("text encoder", "文本编码", 0.150),
        new("refine text", "文本精修", 0.010),
        new("Qwen vision", "参考图编码", 0.020),
        new("audio VAE encoder", "参考音频编码", 0.010),
        new("video VAE encoder", "参考视频编码", 0.010),
        new("AdaLN", "调制预计算", 0.040, "precompute AdaLN"),
        new("load transformer core", "加载 DiT 权重", 0.100),
        new("denoise", "去噪采样", 0.350),
        new("denoise enqueue", "去噪采样", 0.0),
        new("video VAE load", "视频解码器加载", 0.050),
        new("video VAE decode", "视频解码", 0.200),
        new("audio VAE", "音频解码", 0.030),
        new("preview VAE load", "预览解码", 0.005),
        new("FFmpeg", "封装 MP4", 0.015),
        new("save", "写出文件", 0.005)
    };

    /// <summary>
    /// Reference jobs pay for two VAE encodes and two Qwen vision passes before
    /// the encoder even starts, so those stages carry a much larger share here.
    /// Measured on a cold 512x512 / 22-frame run.
    /// </summary>
    private static readonly Stage[] ReferenceStages =
    {
        new("prepare", "准备", 0.005),
        new("tokenizer", "分词", 0.005),
        new("video VAE encoder", "参考视频编码", 0.090),
        new("Qwen vision", "参考图编码", 0.020),
        new("text encoder", "文本编码", 0.200),
        new("refine text", "文本精修", 0.050),
        new("audio VAE encoder", "参考音频编码", 0.010),
        new("AdaLN", "调制预计算", 0.030, "precompute AdaLN"),
        new("load transformer core", "加载 DiT 权重", 0.120),
        new("denoise", "去噪采样", 0.300),
        new("denoise enqueue", "去噪采样", 0.0),
        new("video VAE load", "视频解码器加载", 0.040),
        new("video VAE decode", "视频解码", 0.100),
        new("audio VAE", "音频解码", 0.015),
        new("preview VAE load", "预览解码", 0.005),
        new("FFmpeg", "封装 MP4", 0.010),
        new("save", "写出文件", 0.005)
    };

    public sealed class Plan
    {
        private readonly Dictionary<string, Stage> _byKey =
            new(StringComparer.OrdinalIgnoreCase);

        public Plan(Stage[] stages, double denoiseScale = 1.0)
        {
            /* Denoising is the only stage that scales with the request: frames,
             * steps and layer count multiply its cost, while every other stage
             * is roughly constant.  The tables below are tuned for the default
             * 56-frame / 8-step / 50-layer run, so a 22-frame turbo job needs a
             * smaller share and the 124-frame long take a much larger one. */
            if (denoiseScale != 1.0)
                stages = stages.Select(s => s.Key == "denoise"
                    ? s with { Weight = s.Weight * denoiseScale } : s).ToArray();
            double total = stages.Sum(s => s.Weight);
            TotalWeight = total > 0 ? total : 1.0;
            if (total > 0 && Math.Abs(total - 1.0) > 1e-9)
                stages = stages.Select(s => s with { Weight = s.Weight / total }).ToArray();
            Stages = stages;
            foreach (Stage stage in stages)
            {
                _byKey[stage.Key] = stage;
                foreach (string alias in stage.Aliases) _byKey[alias] = stage;
            }
        }

        public IReadOnlyList<Stage> Stages { get; }

        public double TotalWeight { get; }

        public Stage? Find(string phase) =>
            _byKey.TryGetValue(phase, out Stage? stage) ? stage : null;
    }

    /// <summary>Cost multiplier for denoising relative to the default spec.</summary>
    public static double DenoiseScale(int frames, int steps, int layers)
    {
        double tokens = Math.Max(1, frames) * Math.Max(1, steps) * Math.Max(1, layers);
        double reference = 56.0 * 8 * 50;
        return Math.Clamp(tokens / reference, 0.10, 4.0);
    }

    public static Plan For(string? mode, int frames = 56, int steps = 8, int layers = 50)
    {
        Stage[] stages = string.Equals(mode, GenerationModes.Reference,
            StringComparison.OrdinalIgnoreCase) ? ReferenceStages : SharedStages;
        double scale = DenoiseScale(frames, steps, layers);
        return Math.Abs(scale - 1.0) < 0.01 ? new Plan(stages) : new Plan(stages, scale);
    }

    /// <summary>The default table, for documentation and diagnostics.</summary>
    public static IReadOnlyList<Stage> Stages => SharedStages;

    /// <summary>
    /// Weighted overall progress. <paramref name="reached"/> holds the best
    /// fraction seen per stage key; this method raises it, so the caller only
    /// has to keep the dictionary and clamp nothing.
    /// </summary>
    public static (double Overall, string Label, Stage? Stage) Locate(
        string phase, int completed, int total, IDictionary<string, double> reached,
        Plan plan)
    {
        if (string.IsNullOrWhiteSpace(phase)) return (0, "准备", null);
        Stage? current = plan.Find(phase);
        if (current is not null)
        {
            double fraction = Fraction(completed, total);
            if (!reached.TryGetValue(current.Key, out double best) || fraction > best)
                reached[current.Key] = fraction;
        }
        double done = 0;
        foreach (Stage stage in plan.Stages)
            if (reached.TryGetValue(stage.Key, out double fraction))
                done += stage.Weight * fraction;
        double overall = plan.TotalWeight > 0
            ? Math.Clamp(done / plan.TotalWeight, 0, 0.999) : 0;
        return (overall, current?.Label ?? phase, current);
    }

    private static double Fraction(int completed, int total) =>
        total > 0 ? Math.Clamp((double)completed / total, 0, 1) : 0;
}

/// <summary>
/// Per-job progress bookkeeping: monotonic overall fraction, console lines, ETA.
/// The engine callback runs inline on the worker thread, so everything here is
/// under the job's own lock.
/// </summary>
public sealed class JobProgress
{
    private readonly Dictionary<string, double> _reached = new(StringComparer.OrdinalIgnoreCase);
    private ProgressPlan.Plan _plan = ProgressPlan.For(null);
    private string _phase = string.Empty;
    private string _label = string.Empty;
    private double _overall;
    private double _rate;                 // fraction of the job per second, smoothed
    private double _eta = -1;
    private long _lastLogTicks;
    private double _lastLogOverall = -1;

    public string Phase => _phase;
    public string Label => _label;
    public double Overall => _overall;

    /// <summary>
    /// Remaining seconds, or a negative value when the run is too young to say.
    /// The console line and the page read the same number, so they never
    /// disagree about how long is left.
    /// </summary>
    public double EtaSeconds => _eta;

    /// <summary>
    /// Binds the plan to the request shape. Called once when the job starts, so
    /// the denoise share matches the frame/step/layer count actually requested.
    /// </summary>
    public void Configure(string? mode, int frames, int steps, int layers)
    {
        _plan = ProgressPlan.For(mode, frames, steps, layers);
        _reached.Clear();
        _overall = 0;
        _rate = 0;
        _eta = -1;
    }

    /// <summary>Records one engine report; returns the console line to print, if any.</summary>
    public string? Update(string phase, int completed, int total, long startedTicks)
    {
        (double overall, string label, _) =
            ProgressPlan.Locate(phase, completed, total, _reached, _plan);
        bool newPhase = !string.Equals(phase, _phase, StringComparison.Ordinal);
        _phase = phase;
        _label = label;
        // Monotonic: a stage that reports 50/50 then the next one 0/24 must not
        // pull the bar back.
        if (overall > _overall) _overall = overall;

        long now = Environment.TickCount64;
        double elapsed = startedTicks == 0 ? 0 : (now - startedTicks) / 1000.0;
        // Smooth the rate instead of dividing straight through: a stage whose
        // real cost is larger than its planned share would otherwise make the
        // estimate climb for the whole stage.
        if (elapsed > 2 && _overall > 0.01)
        {
            double sample = _overall / elapsed;
            _rate = _rate > 0 ? _rate * 0.75 + sample * 0.25 : sample;
        }
        if (_rate > 0 && _overall > 0.02)
        {
            double estimate = (1 - _overall) / _rate;
            /* Follow the smoothed rate, but let the estimate fall freely and
             * only creep up: a stage that costs more than its share must not
             * make the number balloon, while a fast final stage has to be able
             * to pull it back down to the truth. */
            _eta = _eta < 0 || estimate < _eta
                ? estimate
                : Math.Min(estimate, _eta * 1.05 + 1.0);
        }

        bool finishedStage = total > 0 && completed >= total;
        bool timeDue = now - _lastLogTicks >= 5000;
        bool stepDue = _overall - _lastLogOverall >= 0.05;
        if (!newPhase && !finishedStage && !timeDue && !stepDue) return null;
        _lastLogTicks = now;
        _lastLogOverall = _overall;

        string detail = total > 0 ? $"{completed}/{total}" : "-";
        string eta = _eta > 0 ? $" · 剩余约 {Format(_eta)}" : string.Empty;
        return $"{label,-8} {detail,-9} {_overall * 100,5:0.0}%  {Format(elapsed)}{eta}";
    }

    /// <summary>Marks every planned stage complete, so the bar lands on 100%.</summary>
    public void Finish()
    {
        foreach (ProgressPlan.Stage stage in _plan.Stages) _reached[stage.Key] = 1;
        _overall = 1;
        _eta = 0;
        _phase = "完成";
        _label = "完成";
    }

    /// <summary>Line for the run summary.</summary>
    public static string Format(double seconds)
    {
        if (seconds < 60) return $"{seconds:0.0}s";
        int minutes = (int)(seconds / 60);
        double rest = seconds - minutes * 60;
        return minutes >= 60
            ? $"{minutes / 60}h{minutes % 60:00}m"
            : $"{minutes}m{rest:00}s";
    }
}
