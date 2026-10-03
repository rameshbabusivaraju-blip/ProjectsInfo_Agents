using ProjectPulse.Web;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();

// One place that knows where the API lives. The timeout is long because the
// free Render host sleeps when idle and can take about a minute to wake up.
var baseUrl = builder.Configuration["ProjectPulseApi:BaseUrl"]
    ?? throw new InvalidOperationException("ProjectPulseApi:BaseUrl is not set.");
builder.Services.AddHttpClient<ProjectPulseClient>(client =>
{
    client.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
    client.Timeout = TimeSpan.FromSeconds(90);

    // The API needs this key on /ask and /files (AGENTS-53). It comes from user secrets or
    // the ProjectPulseApi__ApiKey environment variable, never from a file in the repo.
    var apiKey = builder.Configuration["ProjectPulseApi:ApiKey"];
    if (!string.IsNullOrWhiteSpace(apiKey))
    {
        client.DefaultRequestHeaders.Add("X-API-Key", apiKey);
    }
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}
app.UseStaticFiles();
app.UseRouting();
app.MapRazorPages();

app.Run();
