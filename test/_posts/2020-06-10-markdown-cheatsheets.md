---
title: "Markdown Cheatsheets"
categories: 
    - "markdown"
tags:
    - "jupyter notebook"
    - "markdown"
last_modified_at: 2020-06-10T12:00:00+09:00
---


# Markdown Cheatsheet
This is intended as a quick reference and showcase for Markdown.
For more cheatsheet, check [Adam-P Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)


## Shortcuts

 `escape` key: changes to command mode 
<br>`escape` then `s`: shortcut for "save"
<br>`escape` then `y`: switch to code cell
<br>`escape` then `m`: switch to Markdown cell
<br>`Shift + Enter`: run code cells


## Headers
Put 1-6 hash marks(#) at the beginning of the line

```
# Header 1 
## Header 2 
### Header 3
```
renders as
![header](/assets/images/headers.png)

## Emphasis

+ For bold, wrap the text in two asterisks or underscores (```**bold** or __bold__```) renders as **bold**
+ For italics, wrap the text in one asterisk or underscore (```*italic* or _italic_```) renders as *italic*
+ For strikethrough, wrap the text in two tildes (```~~strikethrough~~```) renders as ~~strikethrough~~


## Code and Syntax Highlighting
1. code
    + start a new line and wrap the text in three backticks.
    + indent each line of the code block with four spaces.

2. Syntax Highlighting
    ``` 
    `sysntax highlighting` 
    ```
    renders as `sysntax highlighting`


## Math expressions

Create math expressions in Markdown cells using LaTeX symbols. 

    $y = mx + b$
renders as $y = mx + b$


## Add images
```
![Alt Text](Url)
```

![My profile](https://avatars3.githubusercontent.com/u/14879156?s=460&u=30c6440b9fb8e96506494a80cb01b1c80039f83e&v=4)


## Colors: 
```
<font color=blue|red|green|pink|yellow>Text</font> 
```
Not all markdown code works within a font tag, so review your colored text carefully!


## Indented quoting
Use a greater than sign (>) and then a space, then type the text. The text is indented and has a gray horizontal line to the left of it until the next carriage return.


## Bullets 
Use the dash sign (- ) with a space after it, or a space, a dash, and a space ( - ), to create a circular bullet. To create a sub bullet, use a tab followed a dash and a space. You can also use an asterisk instead of a dash, and it works the same.


## Internal links
To link to a section, use this code: 
```
[section title](#section-title)
```
For the text in the parentheses, replace spaces and special characters with a hyphen. Make sure to test all the links!
Alternatively, you can add an ID for a section right above the section title. Use this code: 
``` 
<a id="section_ID"></a> 
```
Make sure that the section_ID is unique within the notebook.
Use this code for the link and make sure to test all the links! 
```
[section title](#section_ID)
```


## External links
Use this code and test all links! 
```
[link text](http://url)
```


```python

```
