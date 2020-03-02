# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
import re
import os

TEXT_DIR = os.path.join('..', 'data', 'corpora', 'scraped')

class HsSpider(CrawlSpider):
    name = 'hs'
    allowed_domains = ['hs.fi']
    start_urls = ['https://hs.fi/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        filename = 'hs-2020-03-02.txt'

        paragraphs = response.xpath("//div[@class='body']/span[@class='votsikko']/text() | //div[@class='body']/text()").getall()

        for p in paragraphs:
            p += "\n"
            with open(os.path.join(TEXT_DIR, filename), 'a', encoding="utf-8") as f:
                f.write(p)
