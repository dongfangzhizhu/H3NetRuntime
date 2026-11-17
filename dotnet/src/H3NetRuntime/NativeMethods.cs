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
    /// Locates h3rt.dll / libh3rt.so. Search order:
    /// <list type="number">
    /// <item><c>H3RT_NATIVE_DIR</c></item>
    /// <item>the application base directory (a hand-copied DLL)</item>
    /// <item><c>runtimes/&lt;rid&gt;/native/</c> — the NuGet package layout</item>
    /// <item><c>dist/&lt;rid&gt;/</c> in any parent directory (source tree)</item>
    /// <item>the current working directory</item>
    /// <item>the platform default loader</item>
    /// </list>
    /// When nothing matches the exception lists every path that was tried,
    /// because "DLL not found" with no paths is the least useful error a
    /// packaged runtime can produce.
    /// </summary>
    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != Library) return nint.Zero;
        string file = OperatingSystem.IsWindows() ? "h3rt.dll" : "libh3rt.so";
        string[] rids = RuntimeIdentifiers;
        var tried = new List<string>();

        nint TryPath(string? path)
        {
            if (string.IsNullOrEmpty(path)) return nint.Zero;
            tried.Add(path);
            return TryLoad(path);
        }

        string? explicitDir = Environment.GetEnvironmentVariable("H3RT_NATIVE_DIR");
        nint handle = TryPath(explicitDir is null ? null : Path.Combine(explicitDir, file));
        if (handle != nint.Zero) return handle;

        handle = TryPath(Path.Combine(AppContext.BaseDirectory, file));
        if (handle != nint.Zero) return handle;

        foreach (string rid in rids)
        {
            handle = TryPath(Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", file));
            if (handle != nint.Zero) return handle;
        }

        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        for (int depth = 0; depth < 8 && dir is not null; depth++, dir = dir.Parent)
        {
            foreach (string rid in rids)
            {
                handle = TryPath(Path.Combine(dir.FullName, "dist", rid, file));
                if (handle != nint.Zero) return handle;
            }
        }

        handle = TryPath(Path.Combine(Directory.GetCurrentDirectory(), file));
        if (handle != nint.Zero) return handle;

        try
        {
            if (NativeLibrary.TryLoad(file, assembly, searchPath, out handle)) return handle;
        }
        catch (DllNotFoundException) { }
        catch (BadImageFormatException) { }

        throw new DllNotFoundException(
            $"cannot load the native h3rt runtime ({file}). Tried:{Environment.NewLine}  " +
            string.Join(Environment.NewLine + "  ", tried) + Environment.NewLine +
            "Set H3RT_NATIVE_DIR to the directory holding the native library, or copy it " +
            "next to the application.");
    }

    private static readonly string[] RuntimeIdentifiers = BuildRuntimeIdentifiers();

    private static string[] BuildRuntimeIdentifiers()
    {
        var rids = new List<string>();
        string current = RuntimeInformation.RuntimeIdentifier;
        if (!string.IsNullOrEmpty(current)) rids.Add(current);

        string arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            Architecture.X86 => "x86",
            _ => "x64"
        };
        string family = OperatingSystem.IsWindows() ? "win"
            : OperatingSystem.IsLinux() ? "linux"
            : OperatingSystem.IsMacOS() ? "osx"
            : "win";
        rids.Add(family + "-" + arch);
        return rids.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
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
