---
title: "Blog 개발 환경 구축"
categories: 
    - "blogging"
tags:
    - "blogging"
    - "jekyll"
last_modified_at: 2020-07-19T12:00:00+09:00
---

지금 제가 사용하고 있는 thinkreen 블로그는 `Github.io` 와 `jekyll`을 기반으로 하고있고, `minimal-mistake` 테마를 사용하고 있습니다.
그리고 개발환경은 Git repository를 매번 clone한 후 `Visual Studio Code`를 통해 수정 후 commit/push를 하고 있지만, 자동으로 deploy되도록 변경할 예정입니다.
<br>
<br>

# Posting step
1. write a draft.
2. Run on test server and check grammars/error.
3. Moving a file to `_posts` directory after posting.
4. Commit and push.
5. Check updated posting.
6. Keep updating if needed.


# Structure
아직은 `minimal-mistake` 테마의 구조와 일부 파일들을 그대로 가지고 있지만, 조금씩 수정을 통해 구조를 정리할 예정이고 현재 전반적인 블로구 구조입니다.

```
thinkreen.github.io
├── _data                      # 커스텀 가능한 다양한 데이터 파일 존재 data files for customizing the theme
|  ├── navigation.yml          # 네비게이션바, 사이드바 등이 정의 main navigation links
|  └── ui-text.yml             # 언어별 표시되는 단어들의 정의 text used throughout the theme's UI
├── test                       # 블로그에 포스팅 되기 전 초안 밑 작업 파일들의 저장 및 테스트.
├── _includes                  # includes를 사용하여 가져올 수 있는 파일들. (현재 하나씩 공부중)
|  ├── analytics-providers     # snippets for analytics (Google and custom)
|  ├── comments-providers      # snippets for comments
|  ├── footer                  # custom snippets to add to site footer
|  ├── head                    # custom snippets to add to site head
|  ├── feature_row             # feature row helper
|  ├── gallery                 # image gallery helper
|  ├── group-by-array          # group by array helper for archives
|  ├── nav_list                # navigation list helper
|  ├── toc                     # table of contents helper
|  └── ...
├── _layouts                   # 사이트 전반적인 레이아웃부터 부분적인 레이아웃을 설정
|  ├── archive-taxonomy.html   # tag/category archive for Jekyll Archives plugin
|  ├── archive.html            # archive base
|  ├── categories.html         # archive listing posts grouped by category
|  ├── category.html           # archive listing posts grouped by specific category
|  ├── collection.html         # archive listing documents in a specific collection
|  ├── compress.html           # compresses HTML in pure Liquid
|  ├── default.html            # base for all other layouts
|  ├── home.html               # home page
|  ├── posts.html              # archive listing posts grouped by year
|  ├── search.html             # search page
|  ├── single.html             # single document (post/page/etc)
|  ├── tag.html                # archive listing posts grouped by specific tag
|  ├── tags.html               # archive listing posts grouped by tags
|  └── splash.html             # splash page
├── _sass                      # partial CSS 파일들 저장
├── assets                     # 메인 CSS파일 및 JS, 이미지 파일 저장
|  ├── css
|  |  └── main.scss            # main stylesheet, loads SCSS partials from _sass
|  ├── images                  # image assets for posts/pages/collections/etc.
|  ├── js
|  |  ├── plugins              # jQuery plugins
|  |  ├── vendor               # vendor scripts
|  |  ├── _main.js             # plugin settings and other scripts to load after jQuery
|  |  └── main.min.js          # optimized and concatenated script file loaded before
├── _config.yml                # site configuration
├── Gemfile                    # gem file dependencies
├── index.html                 # paginated home page showing recent posts
└── package.json               # NPM build scripts

```


# jekyll environment
테스트 서버 구동을 위한 jekyll 환경 설정

```
gem install bundler
gem install jekyll

bundle install

# Run test server
bundle exec jekyll serve --host 0.0.0.0 --port 80

# See drafts (all files in /test/)
bundle exec jekyll serve --host 0.0.0.0 --port 80 --test
```
