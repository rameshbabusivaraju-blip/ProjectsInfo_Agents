using System.Net;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace ProjectPulse.Web.Pages;

/// <summary>The question the browser sends to the Ask handler.</summary>
public record AskInput(string? Question);

/// <summary>
/// Home page: the chat screen. The page itself shows the API's /health; the Ask handler is
/// what the browser script calls with each question, and it calls /ask on the API server-side.
/// </summary>
public class IndexModel(ProjectPulseClient api, ILogger<IndexModel> logger) : PageModel
{
    private const int MaxQuestionLength = 1000;

    /// <summary>The API's health answer, or null if the call failed.</summary>
    public HealthResult? Health { get; private set; }

    /// <summary>A plain-English reason when the API could not be reached.</summary>
    public string? ErrorMessage { get; private set; }

    /// <summary>Calls the API's /health once when the page loads.</summary>
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

    /// <summary>
    /// Handles POST /?handler=Ask. Returns the API's answer as JSON, or { error } with a
    /// plain-English reason. The browser never sees the API key or talks to the API itself.
    /// </summary>
    public async Task<IActionResult> OnPostAskAsync([FromBody] AskInput input)
    {
        var question = input.Question?.Trim();
        if (string.IsNullOrEmpty(question) || question.Length > MaxQuestionLength)
        {
            return BadRequest(new { error = $"Please type a question of up to {MaxQuestionLength} characters." });
        }

        try
        {
            var result = await api.AskAsync(question, HttpContext.RequestAborted);
            return new JsonResult(result);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.ServiceUnavailable)
        {
            logger.LogError(ex, "The API refused the request; check the API key setting");
            return StatusCode(502, new { error = "The API refused the request. Check that the API key here matches the one on the API." });
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            logger.LogError(ex, "Ask call to the ProjectPulse API failed");
            return StatusCode(502, new { error = "The ProjectPulse API did not answer. It may still be waking up; try again in a minute." });
        }
    }

    /// <summary>
    /// Handles GET /?handler=File&amp;name=report.xlsx. Fetches the file from the API with the
    /// key and passes it to the browser, so the browser can download it without ever holding the key.
    /// </summary>
    public async Task<IActionResult> OnGetFileAsync(string? name)
    {
        // Only a plain file name is allowed; anything with a folder part is refused.
        if (string.IsNullOrWhiteSpace(name) || name != Path.GetFileName(name))
        {
            return BadRequest();
        }

        try
        {
            var file = await api.DownloadAsync(name, HttpContext.RequestAborted);
            if (file is null)
            {
                return NotFound("That file is no longer available. Ask the question again to create it.");
            }
            return File(file.Content, file.ContentType, file.FileName);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            logger.LogError(ex, "File download from the ProjectPulse API failed");
            return StatusCode(502, "The ProjectPulse API did not answer. Try again in a minute.");
        }
    }
}
