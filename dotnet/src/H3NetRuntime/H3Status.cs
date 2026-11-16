namespace H3NetRuntime;

/// <summary>Status codes returned by the native h3rt ABI.</summary>
public enum H3Status
{
    Ok = 0,
    InvalidArgument = -1,
    InvalidJson = -2,
    Model = -3,
    Device = -4,
    Generate = -5,
    Busy = -6,
    Cancelled = -7,
    OutOfMemory = -8,
    Unsupported = -9,
    Thread = -10
}

/// <summary>Thrown when a native call returns a non-zero status.</summary>
public sealed class H3RuntimeException : Exception
{
    public H3RuntimeException(H3Status status, string message)
        : base($"{message} (h3rt status {(int)status}: {status})")
    {
        Status = status;
    }

    public H3RuntimeException(string message) : base(message)
    {
        Status = H3Status.Generate;
    }

    /// <summary>The native status code that caused the failure.</summary>
    public H3Status Status { get; }
}
