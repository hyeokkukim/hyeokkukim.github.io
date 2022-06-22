---
title: "Web Crawling by Selenium"
categories: 
    - "machine learning"
tags:
    - "machine learning"
    - "python"
    - "crawling"
last_modified_at: 2020-07-01T12:00:00+09:00
---

# Google image download using Selenium
머신 러닝 training을 위한 이미지 크롤링 라이브러리 [google_images_downloads](https://pypi.org/project/google_images_download/) 가 잘 실행되지 않아서 확인해 보았다.<br>
2020년 2월부터 Google images DOM이 image element class="rg_meta notranslate"에서 "rg_i Q4LuWd" 형식으로 변경되면서 더 이상 실행이 되지 않는 문제 대안으로 Selenium을 이용한 크롤링 간편화한 [gids](https://pypi.org/project/gids/) 패키지를 이용하는 방법을 사용하였다.

```
def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')    

def main(args):
    query = "typical face"
    query = query.split()
    query = '+'.join(query)
    url = "https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    soup = get_soup(url, headers)
    for a in soup.find_all("img", {"class": "rg_i"}):
        wget.download(a.attrs["data-iurl"], a.attrs["data-iid"])


if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()
```

# Install Package
```
pip install gids
```

# Usage
## Example code
```python
from gids import builder

config = {
    'driver_path': '/usr/local/bin/chromedriver',
    'headless': True,
    'window-size': '720x480',
    'disable_gpu': False
}

first_item = {
    'keyword': 'White Shark',
    'limit': 10, # The number of images
    'download_context': './data',
    'path': 'animal' # save in ./data/animal/img_01...10
}

second_item = {
    'keyword': 'Whale Shark',
    'limit': 10, # The number of images
    'download_context': './data',
    'path': 'Shark' # save in ./data/plant/img_01...10
}

items = [first_item, second_item]

downloader = builder.build(config)

downloader.download(items)
```
## Result
```

    Loading Pages. This may take a few moments...
    Page Scroll done...
    Start to downloading
    Downloading ...White Shark - [./data/animal/White Shark/img_0]
    Downloading ...White Shark - [./data/animal/White Shark/img_1]
    Downloading ...White Shark - [./data/animal/White Shark/img_2]
    Downloading ...White Shark - [./data/animal/White Shark/img_3]
    Downloading ...White Shark - [./data/animal/White Shark/img_4]
    Downloading ...White Shark - [./data/animal/White Shark/img_5]
    Downloading ...White Shark - [./data/animal/White Shark/img_6]
    Downloading ...White Shark - [./data/animal/White Shark/img_7]
    Downloading ...White Shark - [./data/animal/White Shark/img_8]
    Downloading ...White Shark - [./data/animal/White Shark/img_9]
    White Shark download completed. [Successful count = 10].
    Loading Pages. This may take a few moments...
    Page Scroll done...
    Start to downloading
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_0]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_1]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_2]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_3]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_4]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_5]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_6]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_7]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_8]
    Downloading ...Whale Shark - [./data/Shark/Whale Shark/img_9]
    Whale Shark download completed. [Successful count = 10].
    Total time is 36.451616048812866 seconds.
```

# Troubleshooting
아래와 같이 에러가 나올 때 chromedriver파일이 있는 디렉토리로 PATH를 설정해줌.

No found chromedriver in this environment.
Install on your machine. exception: Message: 'chromedriver' executable needs to be in PATH. Please see https://sites.google.com/a/chromium.org/chromedriver/home

```
export PATH=$PATH:/usr/local/bin/chromedriver
```
