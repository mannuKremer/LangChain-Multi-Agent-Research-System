from src.tools.tools import web_search, scrape_url



output = web_search.invoke("Latest news about AI research")
# print(output)

scrapeUelRes = scrape_url.invoke("https://ai.cornell.edu/category/featured")
print(scrapeUelRes)