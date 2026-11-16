using System.IO.Compression;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Minimal PNG encoder used for live previews and history posters. It
/// downscales with nearest-neighbour (cheap, no dependencies) and compresses
/// with zlib. This runs on a preview thread, never on the engine callback.
/// </summary>
public static class PngEncoder
{
    private static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>Encodes RGB24 (stride may be larger than width*3) as a PNG.</summary>
    public static byte[] Encode(ReadOnlySpan<byte> rgb, int width, int height, int stride,
                                int targetWidth)
    {
        if (width <= 0 || height <= 0) return Array.Empty<byte>();
        int outWidth = Math.Clamp(targetWidth, 1, width);
        int outHeight = Math.Max(1, (int)Math.Round((double)height * outWidth / width));

        int rowBytes = outWidth * 3;
        byte[] raw = new byte[(rowBytes + 1) * outHeight];
        int offset = 0;
        for (int y = 0; y < outHeight; y++)
        {
            raw[offset++] = 0; // filter: none
            int sourceY = (int)((long)y * height / outHeight);
            int sourceRow = sourceY * stride;
            for (int x = 0; x < outWidth; x++)
            {
                int sourceX = (int)((long)x * width / outWidth) * 3;
                int source = sourceRow + sourceX;
                raw[offset++] = rgb[source];
                raw[offset++] = rgb[source + 1];
                raw[offset++] = rgb[source + 2];
            }
        }

        using var compressed = new MemoryStream(raw.Length / 2 + 64);
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
            zlib.Write(raw, 0, raw.Length);

        using var png = new MemoryStream((int)compressed.Length + 128);
        png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
        Span<byte> header = stackalloc byte[13];
        WriteBigEndian(header, (uint)outWidth);
        WriteBigEndian(header[4..], (uint)outHeight);
        header[8] = 8;  // bit depth
        header[9] = 2;  // colour type: RGB
        header[10] = 0; // compression
        header[11] = 0; // filter
        header[12] = 0; // interlace
        WriteChunk(png, "IHDR", header);
        WriteChunk(png, "IDAT", compressed.GetBuffer().AsSpan(0, (int)compressed.Length));
        WriteChunk(png, "IEND", ReadOnlySpan<byte>.Empty);
        return png.ToArray();
    }

    private static void WriteChunk(Stream output, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[4];
        WriteBigEndian(length, (uint)data.Length);
        output.Write(length);

        Span<byte> typeBytes = stackalloc byte[4];
        for (int i = 0; i < 4; i++) typeBytes[i] = (byte)type[i];
        output.Write(typeBytes);
        output.Write(data);

        uint crc = 0xFFFFFFFFu;
        crc = UpdateCrc(crc, typeBytes);
        crc = UpdateCrc(crc, data);
        crc ^= 0xFFFFFFFFu;
        Span<byte> crcBytes = stackalloc byte[4];
        WriteBigEndian(crcBytes, crc);
        output.Write(crcBytes);
    }

    private static void WriteBigEndian(Span<byte> destination, uint value)
    {
        destination[0] = (byte)(value >> 24);
        destination[1] = (byte)(value >> 16);
        destination[2] = (byte)(value >> 8);
        destination[3] = (byte)value;
    }

    private static uint UpdateCrc(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (byte value in data)
            crc = CrcTable[(crc ^ value) & 0xFF] ^ (crc >> 8);
        return crc;
    }

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }
}
