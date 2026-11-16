using System.Runtime.InteropServices;

namespace H3NetRuntime;

/// <summary>Owns a native h3rt_handle and releases it exactly once.</summary>
internal sealed class H3RuntimeHandle : SafeHandle
{
    public H3RuntimeHandle(nint handle) : base(nint.Zero, ownsHandle: true)
    {
        SetHandle(handle);
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.Destroy(handle);
        return true;
    }
}
