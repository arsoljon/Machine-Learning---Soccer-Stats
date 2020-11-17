import requests
import time
import re
from bs4 import BeautifulSoup
import pandas as pd

def get_fifa_data():
    url = "https://www.fifaindex.com/teams/fifa20_358/?league=13&order=desc"
    r = requests.get(url)
    #check if the site being used sends a 200 status code. Verifying it can be webscraped.
    print(r.status_code)

    #break down site html and siphon out team data.
    soup = BeautifulSoup(r.text, "html.parser")
    soup = soup.findAll('tbody')
    soup = soup[0].text
    soup = soup.replace('\n', '')
    #remove Premier league string from the results
    delete_league = "Premier League"
    soup = soup.replace(delete_league, " ")
    #separate the names and numbers from the data
    list_of_names = re.findall('\D+', soup)
    list_of_nums = re.findall("\d{8}", soup)

    #dictionary to hold our new data
    data = {"Squads": []}
    #organize names and averages into the list of data
    i = 0
    for n in list_of_names:
        _att = list_of_nums[i][:2]
        _mid = list_of_nums[i][2:4]
        _def = list_of_nums[i][4:6]
        _ovr = list_of_nums[i][6:8]
        data["Squads"].append({"Squad_name": n, "Details" : {"ATT": _att, "MID": _mid, "DEF": _def, "OVR": _ovr}})
        i += 1

    #normalize our team data into a json dataframe so we can make a easy to read csv file
    df = pd.json_normalize(data["Squads"])
    df.to_csv("../Data/team_data_fifa.csv", index = False)

    #avoiid being flagged as a spammer from site
    time.sleep(1)
    return df


def get_possession_data():
    path = "../PossesionEPL20192020(2).xlsx"
    df = pd.DataFrame(pd.read_excel(path))
    df.to_csv("../Data/team_poss_data.csv", index = False)
    return df

def get_epl_data():
    path = "../EPLTable20192020.xlsx"
    df = pd.DataFrame(pd.read_excel(path))
    df.to_csv("../Data/team_elp_data.csv", index = False)
    return df

get_epl_data()
get_fifa_data()
get_possession_data()

