#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import requests
from requests.models import Response
import bs4
from typing import Dict


class NlpGutenburgLoader:
    def __init__(self, top_books_category: str = "Top 100 EBooks yesterday"):
        self.frequently_download_url = "https://www.gutenberg.org/browse/scores/top"
        self.top_books_category = top_books_category

    @staticmethod
    def get_request(url: str) -> Response:
        req = requests.get(url)
        try:
            req.raise_for_status()
        except Exception as exc:
            print(f"There was a problem: {exc}")
        return req
    
    def get_top_book_names_urls(self) -> Dict[str, str]:
        gutenberg_req = NlpGutenburgLoader.get_request(self.frequently_download_url)
        gutenberg_soup = bs4.BeautifulSoup(gutenberg_req.text)

        self.top_book_names_urls = dict()

        for title, element in zip(gutenberg_soup.find_all("h2"), gutenberg_soup.find_all('ol')):
            if title.text == self.top_books_category:
                print(f"Getting {title.text}'s names and URLs")
                for line in element.find_all("li"):
                    book_name = line.get_text()
                    book_id = line.a["href"].split("/")[-1]
                    book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                    self.top_book_names_urls[book_name] = book_url
        
        return self.top_book_names_urls
    

    def load_top_books(self):
        for book_name, book_url in iter(self.top_book_names_urls.items()):
            book_title = book_name.rsplit(" by", maxsplit=1)[0]
            print(f"Book title: {book_title}")
            
            req = NlpGutenburgLoader.get_request(book_url)
            book_text = req.text
            print("Starting truncating book text.")

            # Truncate head substring.
            truncate_head_substr = f"*** START OF THE PROJECT GUTENBERG EBOOK {book_title.upper()} ***"
            book_text = book_text.split(truncate_head_substr)[1]
            print(f"Truncating head substring is done!")

            # Truncate tail substring.
            truncate_tail_substr = f"*** END OF THE PROJECT GUTENBERG EBOOK {book_title.upper()} ***"
            book_text = book_text.split(truncate_tail_substr)[0]
            print(f"Truncating tail substring is done!")
            
            # TODO: Preprocessing text.
            break

