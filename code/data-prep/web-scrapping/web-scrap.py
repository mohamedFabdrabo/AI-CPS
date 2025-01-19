import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

Leagues_IDs = ['eng.1', 'esp.1', 'ger.1', 'ita.1', 'fra.1','sco.1','por.1','ksa.1','tur.1','gre.1','aut.1','jpn.1','bra.1'
               , 'eng.2', 'eng.3', 'esp.2', 'esp.3', 'fra.2', 'fra.3', 'ger.2', 'ger.3', 'ita.2', 'ita.3']
# Base URL template
base_url = "https://www.espn.co.uk/football/table/_/league/{league_id}/season/{year}/league"

session = requests.Session()
# Use headers with the session
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.co.uk/"
})

# Function to scrape a single league and season
# Function to scrape a single league and season
def scrape_league_table(league_id, year):
    url = base_url.format(league_id=league_id, year=year)
    
    response = session.get(url)
    if response.status_code != 200:
        print(f"\nFailed to fetch data for league {league_id}, season {year}. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    # find all tables from the html response
    tables = soup.find_all('table')
    # Find the league table
    if not tables:
        print(f"\nNo table found for league {league_id}, season {year}")
        return None

    # get the name of teams from table[0] in the HTML response
    teams_column = []
    for row in tables[0].find("tbody").find_all("tr"):
        teams_column.append([cell.text.strip() for cell in row.find_all("td")])

    # get the standings table data from table[1] in the HTML response

    # Extract headers
    headers = [header.text for header in tables[1].find("thead").find_all("th")]
    # Extract rows
    rows = []
    for row in tables[1].find("tbody").find_all("tr"):
        rows.append([cell.text.strip() for cell in row.find_all("td")])

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)
    df['team_name'] = teams_column
    df["League"] = league_id
    df["Season"] = year
    
    new_order = ['League', 'Season','team_name', 'GP', 'W', 'D', 'L', 'F', 'A', 'GD', 'P']
    return df[new_order]

# Loop through leagues and seasons
start_season = 2004
end_season = 2024
all_data = []

for league_id in Leagues_IDs:
    for year in range(start_season, end_season + 1):
        print(f"Scraping league {league_id}, season {year} ... ... ...", end = ' ')
        try:
            df = scrape_league_table(league_id, year)
            if df is not None or df.shape[1] != 11:
                all_data.append(df)
                print(f"Added Data rows: {df.shape[0]}")
        except:
            print(f'Skipped scrapping {league_id}, season {year} : Skipped')
        time.sleep(2)  # To avoid overloading the server

# Combine all data into a single DataFrame
# TODO: edit to save in the required file system
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("league_tables.csv", index=False)
    print("Data saved to 'league_tables.csv'.")
else:
    print("No data was scraped.")
