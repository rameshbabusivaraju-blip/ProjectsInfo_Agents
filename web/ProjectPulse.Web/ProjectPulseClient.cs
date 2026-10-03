using System.Text.Json.Serialization;

namespace ProjectPulse.Web;

/// <summary>What the API's /health route returns: a status word and the deployed version.</summary>
public record HealthResult(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("version")] string Version);

/// <summary>
/// Calls the ProjectPulse FastAPI backend from the server side.
/// This is a typed client: IHttpClientFactory builds the HttpClient and hands it in,
/// so the base address and timeout are set once in Program.cs.
/// </summary>
public class ProjectPulseClient(HttpClient http)
{
    /// <summary>Calls GET /health. Throws HttpRequestException if the API cannot be reached.</summary>
    public async Task<HealthResult?> GetHealthAsync(CancellationToken cancellationToken = default)
    {
        return await http.GetFromJsonAsync<HealthResult>("health", cancellationToken);
    }
}
