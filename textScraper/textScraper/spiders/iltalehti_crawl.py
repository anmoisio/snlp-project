# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import re
import os

# TEXT_DIR = os.path.join('..', 'data', 'corpora', 'scraped', 'iltalehti2')
TEXT_DIR = os.path.join('..')

class IltalehtiSpider(CrawlSpider):
    name = 'iltalehti'
    allowed_domains = ['iltalehti.fi']
    start_urls = ['https://www.iltalehti.fi/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        filename = response.url.split("/")[-2] + '-iltalehti-2020-03-25' + '.txt'

        # scrape paragraphs <p></p> from inside class="article-body"
        paragraphs = response.xpath('//div[@class="article-body"]/p/text()').getall()

        for p in paragraphs:
            p += " " # space between paragraphs
            with open(os.path.join(TEXT_DIR, filename), 'a', encoding="utf-8") as f:
                f.write(p)
        if paragraphs:
            with open(os.path.join(TEXT_DIR, filename), 'a', encoding="utf-8") as f:
                f.write("\n") # one document per line
