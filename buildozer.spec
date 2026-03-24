[app]

# (str) Title of your application
title = Portfolio Optimizer

# (str) Package name
package.name = portfolio_optimizer

# (str) Package domain (needed for android/ios packaging)
package.domain = org.kumoh

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,ttf,json

# (list) List of directory to exclude
source.exclude_dirs = tests, bin, venv, .buildozer, pages, __pycache__

# (str) Application versioning
version = 1.0

# (list) Application requirements
requirements = python3,kivy==2.3.0,kivymd==1.2.0,pillow,numpy==1.26.4,pandas==2.1.4,matplotlib,requests,certifi,charset-normalizer,idna,urllib3,yfinance,lxml,beautifulsoup4,multitasking,peewee,appdirs,frozendict,pytz,six,cycler,kiwisolver,pyparsing,python-dateutil
