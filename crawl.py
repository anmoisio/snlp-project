import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

import re


class MySpider(CrawlSpider):
    name = 'all'
    allowed_domains = ['iltalehti.fi']
    start_urls = ['https://www.iltalehti.fi/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        filename = response.url.split("/")[-2] + '.html-test5'

        body = str(response.body, encoding='utf-8')
        if 'class="media-source"' in body:
            i = body.find('class="media-source"') + len('class="media-source"') + 1
            

            body_2 = body[i:]

            i = body_2.find('<p>')
            body_3 = body_2[i:]
            i = body_3.find('<div class=')
            

            body_4 = body_3[:i] + "\n\n"

            body_4 = re.sub('<[^>]+>', '', body_4)
            
            with open(filename, 'a') as f:
                f.write(body_4)