using System.Reflection;
using System.Runtime.InteropServices;

namespace H3NetRuntime;

/// <summary>
/// Raw C ABI bindings. Nothing here is public: use <see cref="H3Runtime"/>.
/// All entry points use cdecl and UTF-8 strings, matching include/h3rt.h.
/// </summary>
internal static class NativeMethods
{
    private const string Library = "h3rt";

    static NativeMethods()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, ResolveLibrary);
    }

    /// <summary>
    /// Locates h3rt.dll / libh3rt.so. Search order: H3RT_NATIVE_DIR,
    /// the app base directory, any dist/&lt;rid&gt; folder above it, the CWD,
    /// then the platform default loader.
    /// </summary>
    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != Library) return nint.Zero;
        string file = OperatingSystem.IsWindows() ? "h3rt.dll" : "libh3rt.so";
        string rid = OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";

        string? explicitDir = Environment.GetEnvironmentVariable("H3RT_NATIVE_DIR");
        if (!string.IsNullOrEmpty(explicitDir))
        {
            nint fromEnv = TryLoad(Path.Combine(explicitDir, file));
            if (fromEnv != nint.Zero) return fromEnv;
        }

        nint handle = TryLoad(Path.Combine(AppContext.BaseDirectory, file));
        if (handle != nint.Zero) return handle;

        // Development layout: <repo>/dotnet/<project>/bin/<cfg>/<tfm> -> <repo>/dist/<rid>.
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        for (int depth = 0; depth < 8 && dir is not null; depth++, dir = dir.Parent)
        {
            handle = TryLoad(Path.Combine(dir.FullName, "dist", rid, file));
            if (handle != nint.Zero) return handle;
        }

        handle = TryLoad(Path.Combine(Directory.GetCurrentDirectory(), file));
        if (handle != nint.Zero) return handle;
        return TryLoad(file);
    }

    private static nint TryLoad(string path)
    {
        try
        {
            return NativeLibrary.TryLoad(path, out nint handle) ? handle : nint.Zero;
        }
        catch (DllNotFoundException) { return nint.Zero; }
        catch (BadImageFormatException) { return nint.Zero; }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int ProgressCallback(nint phaseUtf8, int completed, int total, nint opaque);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int FrameCallback(nint frame, nint opaque);

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeFrame
    {
        public int Width;
        public int Height;
        public int Stride;
        public nint Rgb;
        public int FrameIndex;
        public int FrameCount;
        public int DenoiseStep;
        public int DenoiseSteps;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeCallbacks
    {
        public nint OnProgress;
        public nint OnFrame;
        public nint Opaque;
    }

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_version")]
    internal static extern nint Version();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_abi_version")]
    internal static extern int AbiVersion();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_status_text")]
    internal static extern nint StatusText(int status);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_build_info")]
    internal static extern nint BuildInfo();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_free_string")]
    internal static extern void FreeString(nint text);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_create")]
    internal static extern int Create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? optionsJson,
        out nint handle, nint errorBuffer, nuint errorSize);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_destroy")]
    internal static extern void Destroy(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_last_error")]
    internal static extern nint LastError(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_device_info")]
    internal static extern nint DeviceInfo(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_model_info")]
    internal static extern nint ModelInfo(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_cache_info")]
    internal static extern nint CacheInfo(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_cache_clear")]
    internal static extern int CacheClear(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_set_text_cache_dir")]
    internal static extern int SetTextCacheDir(nint handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string dir);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_preload")]
    internal static extern int Preload(nint handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        nint errorBuffer, nuint errorSize);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_generate")]
    internal static extern int Generate(nint handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        ref NativeCallbacks callbacks, out nint outJson);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_cancel")]
    internal static extern int Cancel(nint handle);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl, EntryPoint = "h3rt_is_cancelled")]
    internal static extern int IsCancelled(nint handle);

    /// <summary>Reads a borrowed (not owned) UTF-8 pointer.</summary>
    internal static string ReadString(nint pointer) =>
        pointer == nint.Zero ? string.Empty : Marshal.PtrToStringUTF8(pointer) ?? string.Empty;

    /// <summary>Copies an owned UTF-8 string and releases the native buffer.</summary>
    internal static string? TakeString(nint pointer)
    {
        if (pointer == nint.Zero) return null;
        try { return Marshal.PtrToStringUTF8(pointer); }
        finally { FreeString(pointer); }
    }
}
