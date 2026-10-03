using Microsoft.AspNetCore.Mvc.RazorPages;

namespace ProjectPulse.Web.Pages;

/// <summary>Home page. For now it only proves the .NET to Python call works by showing /health.</summary>
public class IndexModel(ProjectPulseClient api, ILogger<IndexModel> logger) : PageModel
{
    /// <summary>The API's health answer, or null if the call failed.</summary>
    public HealthResult? Health { get; private set; }

    /// <summary>A plain-English reason when the API could not be reached.</summary>
    public string? ErrorMessage { get; private set; }

    /// <summary>Calls the API once when the page loads.</summary>
    public async Task OnGetAsync()
    {
        try
        {
            Health = await api.GetHealthAsync(HttpContext.RequestAborted);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            logger.LogWarning(ex, "Health call to the ProjectPulse API failed");
            ErrorMessage = "The ProjectPulse API could not be reached. Check that it is running and that ProjectPulseApi:BaseUrl is correct.";
        }
    }
}
