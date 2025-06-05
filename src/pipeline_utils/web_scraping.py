"""
Steam Web Scraping Module
Original module integrated into the Final Pipeline for self-containment
"""

import pandas as pd
import requests
import datetime
import json
import re
import sys
from cleantext import clean

try:
    from tqdm.notebook import tqdm
    from IPython.display import clear_output
    IN_NOTEBOOK = True
except ImportError:
    from tqdm import tqdm
    IN_NOTEBOOK = False

from langdetect import detect
from bs4 import BeautifulSoup

##########################################
##########################################


class scrape_steam_game:
    """
        def:
            the object aims to srcape the game reviews and guides given a steam game id
        
        Functions:
            Private:
                get_user_reviews()
                json_scrape()
                table_scrape()

                get_section_text()
                get_player_guides()

                fix_mojibake()
                translate_emojis_to_text()
                readable_description()

            
            Public:
                scrape_review_info()
                scrape_player_guides()

    """

    def __init__(self, game_id):
        self.game_id = game_id
        self.params = {
            "json": 1,
            "language" : 'english',
            "cursor": "*",
            "num_per_page": 100,   # retrive n reviews per request
            "filter": "recent"
        }

        
    
    ##########################################
    ##########################################

    # Scrape reviews

    # Private Function
    def __get_user_reviews(self, game_id, params):
        """
            def:
                scrape player reviews from a game
            
            parameters:
                game_id: a id of a game from steam
                params: setting of web scraping
            
            return:
                player_reviews in json format

        """
        user_review_url = f'https://store.steampowered.com/appreviews/{game_id}'
        # print(f"Requesting: {user_review_url}")
        # print(f"With params: {params}")
        # sys.stdout.flush()
        
        req_user_review = requests.get(
            user_review_url,
            params = params,
            timeout=30  # Add timeout
        )
        
        # print(f"Response status: {req_user_review.status_code}")
        # sys.stdout.flush()

        if req_user_review.status_code != 200:
            print(f'fail to get reponse. Status Code: {req_user_review.status_code}')
            return {"success": 2}
        
        try:
            players_reviews = req_user_review.json()
        except:
            return {"sucess": 2}
        
        return players_reviews

    
    def __json_scrape(self, game_id, params):
        player_reviews = []
   
        while True:
            reviews = self.__get_user_reviews(game_id = game_id, 
                                        params = params)

            # cannot be scraped
            if reviews["success"] != 1:
                print("Not a success")
                break
            
            if reviews["query_summary"]["num_reviews"] == 0:
                break
            
            review_list = reviews["reviews"]
            
            # iterate the review to check if they are in english
            for review in review_list:
                if self.__en_classifier(review["review"]):
                    # fix issues such as correct Canâ€™t to Can't and remove ' from string
                    review = review["review"].encode('cp1252', errors='replace').decode('utf-8', errors='replace').replace("'", "")
                    cleaned_review = self.__review_cleaning(review)
                    if cleaned_review != "":
                        player_reviews += [review]
            
            try:
                cursor = reviews["cursor"]
            except Exception as e:
                cusor = ''

            if not cursor:
                break
            
            params["cursor"] = cursor

        return player_reviews

    def __table_scrape(self, game_id, params):
        player_review_df = pd.DataFrame(columns=[
        "playtime_forever", "num_games_owned", "num_reviews",
        "votes_up", "votes_funny", "weighted_vote_score", 
        "comment_count", "steam_purchase",
        "written_during_early_access", "primarily_steam_deck",
        "timestamp_created", "review"
    ])

        # Initialize progress tracking
        total_scraped = 0
        pbar = None

        # print("Starting scraping loop...")
        # sys.stdout.flush()
        request_count = 0
        
        while True:
            request_count += 1
            # print(f"Making API request #{request_count}...")
            # sys.stdout.flush()
            
            reviews = self.__get_user_reviews(game_id = game_id, 
                                        params = params)
            
            # print(f"API request #{request_count} completed")
            # sys.stdout.flush()
            
            # cannot be scraped
            if reviews["success"] != 1:
                print(f"API request failed - success code: {reviews.get('success', 'unknown')}")
                sys.stdout.flush()
                break
            
            if reviews["query_summary"]["num_reviews"] == 0:
                # Update progress bar to show completion when Steam returns no more reviews
                if pbar and pbar.n < pbar.total:
                    pbar.total = pbar.n  # Set total to what we actually processed
                    pbar.set_description("Scraping complete")
                    pbar.refresh()
                    sys.stdout.flush()
                break

            # Initialize progress bar with total review count on first iteration
            if pbar is None:
                # print(f"Query summary keys: {list(reviews['query_summary'].keys())}")
                # sys.stdout.flush()
                if "total_reviews" in reviews["query_summary"]:
                    total_reviews = reviews["query_summary"]["total_reviews"]
                    # print(f"Total reviews available: {total_reviews}")
                    # sys.stdout.flush()
                    pbar = tqdm(total=total_reviews, desc="Scraping reviews", unit="reviews")
                else:
                    print("No total_reviews found, using indeterminate progress bar")
                    sys.stdout.flush()
                    pbar = tqdm(desc="Scraping reviews", unit="reviews")

            for review in reviews["reviews"]:

                # check if the review is english
                if self.__en_classifier(review["review"]):
                    text_review = review["review"].encode('cp1252', errors='replace').decode('utf-8', errors='replace').replace("'", "")
                    cleaned_review = self.__review_cleaning(text_review)
                    if cleaned_review == "":
                        continue
                
                    # Get the post time
                    time_stamp = review["timestamp_created"]
                    human_readable_date = datetime.datetime.fromtimestamp(time_stamp, tz=datetime.timezone.utc)
                    formatted_date = human_readable_date.strftime('%Y-%m-%d %H:%M:%S')

                    player_review_df.loc[len(player_review_df)] = {
                        "playtime_forever": review["author"]["playtime_forever"],
                        "num_games_owned": review['author']['num_games_owned'],
                        "num_reviews": review['author']['num_reviews'],
                        "votes_up": review['votes_up'],
                        "votes_funny": review['votes_funny'],
                        "weighted_vote_score": review['weighted_vote_score'],
                        "comment_count": review['comment_count'],
                        "steam_purchase": review['steam_purchase'],
                        "written_during_early_access": review['written_during_early_access'],
                        "primarily_steam_deck": review['primarily_steam_deck'],
                        "timestamp_created": formatted_date,
                        "review": cleaned_review
                    }
                
                # Update progress bar
                total_scraped += 1
                if pbar:
                    pbar.update(1)
            
            try:
                cursor = reviews["cursor"]
            except Exception as e:
                cusor = ''

            if not cursor:
                print("Reached the end of all comments.")
                # Update progress bar to show completion even if we didn't reach the estimated total
                if pbar and pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                    pbar.set_description("Scraping complete (Steam API ended early)")
                    pbar.refresh()  # Force progress bar update
                    sys.stdout.flush()
                break
            
            params["cursor"] = cursor

        # Close progress bar
        if pbar:
            pbar.close()

        # Show filtering results
        print(f"\n📊 Filtering results:")
        print(f"   Total reviews processed: {total_scraped}")
        print(f"   English reviews after filtering: {len(player_review_df)}")
        if total_scraped > 0:
            filter_rate = (total_scraped - len(player_review_df)) / total_scraped * 100
            print(f"   Filtered out: {total_scraped - len(player_review_df)} reviews ({filter_rate:.1f}%)")

        return player_review_df
    
    
    ##########################################
    ##########################################

    # Scrape Guides
    
    def __get_section_text(self, sections):
        """
            def: 
                get text from each section from html

            parameters:
                sections: sections in html

            return:
                guide: text from the page
        """
        guide = ""

        # Iterate each section to get text
        for section in sections:
                    title = section.find("div", class_="subSectionTitle")
                    content = section.find("div", class_="subSectionDesc")

                    title_text = title.get_text(strip=True) if title else "No Title"
                    content_text = content.get_text(separator="\n", strip=True) if content else "No Content"

                    if content_text.replace(" ", "") != "":
                        section_title = f"Section Title: {title_text}\n"
                        content = f"Content: \n{content_text}"
                        guide += section_title + content + "\n"
                    else:
                        break

        return guide

    def __get_player_guides(self):
        """
            def:
                get guides of the game for chatbot
            
            parameters:
                game_id: the id of the game from steam

            return:
                guides: a list of guides from the game on steam
        """
        link = f'https://steamcommunity.com/app/{self.game_id}/guides/?browsefilter=trend&filetype=11&requiredtags%5B0%5D=english&p=1'

        guides = []
        
        while True:
            # request access
            req_game_guide = requests.get(link)

            # Get html
            guide_soup = BeautifulSoup(req_game_guide.text, 'html.parser')

            current_page_num = int(link[-1])

            # Get links of guide
            hrefs = []
            for a_tag in guide_soup.find_all('a', href=True):
                if a_tag.find('div', class_='workshopItem'):
                    hrefs.append(a_tag['href'])
            
            # Get every guides on the page
            for href in hrefs:
                req_player_guide = requests.get(href)
                player_guide_soup = BeautifulSoup(req_player_guide.text, "html.parser")
                sections = player_guide_soup.find_all("div", class_ = "detailBox")

                # Get text
                guide = self.__get_section_text(sections)

                # append guide to a list
                if guide != "":
                    guides.append(guide)

            # find the cursor
            page_cursor = guide_soup.find_all('a', class_ = "pagebtn", href=True)
            next_page_num = int(page_cursor[-1]["href"][-1])


            # if the next page number is less than current page hum break the while loop 
            # otherwise, obtain the link of next page
            if next_page_num < current_page_num: break 
            else: link = page_cursor[-1]["href"]
        
        return guides
    
    ##########################################
    ##########################################

    # clean review

    def __review_cleaning(self, review):
        """
            def: clean latin letters or misread text
               return an empty string if less than 3 words after removing notations  
        """
        # Clean the review first
        cleaned_text = clean(review)
        
        # Simple tokenization (replaces gensim.simple_preprocess)
        # Extract words (letters only), convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
        
        if len(words) >= 3:
            return " ".join(words)
        
        return ""
        
    ##########################################
    ##########################################

    # English Classifier

    def __en_classifier(self, review):
        """
            def:
                classify if a review is in english
            
            parameters:
                review: text data
            
            return:
                a boolean value
        """
        try:
            language = detect(review)
            return language == "en"
        except Exception:
            return False


    ##########################################
    ##########################################


    # Public function
    def scrape_review_info(self):
        """
            def: 
                a scrape reviews    given a game id on Steam
            
            return:
                returns a list of all player review infomation in tabl;e format from table_scrape() 
        """
        # return self.__json_scrape(game_id = self.game_id, 
        #                 params = self.params)

        return self.__table_scrape(game_id = self.game_id,
                            params = self.params)
    
    # Public Function
    def scrape_guides(self):
        """
            def: 
                a scrape guides given a game id on Steam
            
            return:
                returns a list of guide of a steam game
        """
        return self.__get_player_guides()