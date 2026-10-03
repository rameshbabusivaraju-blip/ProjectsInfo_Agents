using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ProjectPulse.Web;

/// <summary>What the API's /health route returns: a status word and the deployed version.</summary>
public record HealthResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("version")] string Version);

/// <summary>One document passage an answer drew on (AGENTS-74 on the API side).</summary>
public record AnswerSource(
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("url")] string? Url,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("score")] double Score);

/// <summary>
/// What POST /ask returns. Rows are the data rows behind the answer, as name/value pairs;
/// FileUrl is set only when the agent wrote a file such as an Excel export.
/// </summary>
public record AskResult(
    [property: JsonPropertyName("question")] string Question,
    [property: JsonPropertyName("answer")] string Answer,
    [property: JsonPropertyName("metric_key")] string MetricKey,
    [property: JsonPropertyName("rows")] List<Dictionary<string, JsonElement>> Rows,
    [property: JsonPropertyName("sources")] List<AnswerSource> Sources,
    [property: JsonPropertyName("file_url")] string? FileUrl);

/// <summary>A file fetched from the API: its bytes, content type and name.</summary>
public record FileDownload(byte[] Content, string ContentType, string FileName);

/// <summary>
/// Calls the ProjectPulse FastAPI backend from the server side.
/// This is a typed client: IHttpClientFactory builds the HttpClient and hands it in,
/// so the base address, timeout and API key header are set once in Program.cs.
/// </summary>
public class ProjectPulseClient(HttpClient http)
{
    /// <summary>Calls GET /health. Throws HttpRequestException if the API cannot be reached.</summary>
    public async Task<HealthResult?> GetHealthAsync(CancellationToken cancellationToken = default)
    {
        return await http.GetFromJsonAsync<HealthResult>("health", cancellationToken);
    }

    /// <summary>
    /// Calls POST /ask with the question. Throws HttpRequestException if the API cannot be
    /// reached or answers with an error; its StatusCode says which (401 means a wrong key).
    /// </summary>
    public async Task<AskResult> AskAsync(string question, CancellationToken cancellationToken = default)
    {
        using var response = await http.PostAsJsonAsync("ask", new { question }, cancellationToken);
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<AskResult>(cancellationToken);
        return result ?? throw new HttpRequestException("The API returned an empty answer.");
    }

    /// <summary>
    /// Downloads a file the agent wrote (GET /files/{name}). Returns null if the API has no such
    /// file, which happens when the host restarted and cleared its exports folder.
    /// </summary>
    public async Task<FileDownload?> DownloadAsync(string fileName, CancellationToken cancellationToken = default)
    {
        using var response = await http.GetAsync($"files/{Uri.EscapeDataString(fileName)}", cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            return null;
        }
        response.EnsureSuccessStatusCode();
        var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        var contentType = response.Content.Headers.ContentType?.ToString() ?? "application/octet-stream";
        return new FileDownload(bytes, contentType, fileName);
    }
}
