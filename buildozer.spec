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
requirements = python3,kivy==2.3.0,kivymd==1.2.0,pillow,numpy,pandas==1.5.3,matplotlib,requests,certifi,charset-normalizer,idna,urllib3,yfinance,multitasking,peewee,appdirs,frozendict,pytz,six,cycler,kiwisolver,pyparsing,python-dateutil

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (list) Supported orientations
orientation = portrait

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Permissions
android.permissions = android.permission.INTERNET, android.permission.ACCESS_NETWORK_STATE

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK / AAB will support.
android.minapi = 24

# (str) Android NDK version to use
android.ndk = 27c

# (str) Android SDK / NDK paths
#android.sdk_path =
#android.ndk_path =

# (int) Android NDK API to use. This is the minimum API your app will support.
android.ndk_api = 24

# (list) The Android archs to build for
android.archs = arm64-v8a

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (string) Presplash background color
android.presplash_color = #FFFFFF

# (bool) If True, then automatically accept SDK license agreements.
android.accept_sdk_license = True

# (str) The format used to package the app for debug mode
android.debug_artifact = apk

# change the major version of python used by the app
osx.python_version = 3

# Kivy version to use
osx.kivy_version = 2.3.0

#
# Python for android (p4a) specific
#

# (str) python-for-android branch to use, defaults to master
p4a.branch = develop

# (str) Bootstrap to use for android builds
p4a.bootstrap = sdl2
p4a.local_recipes = ./p4a-recipes
#
# iOS specific
#

ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
ios.ios_deploy_url = https://github.com/phonegap/ios-deploy
ios.ios_deploy_branch = 1.10.0
ios.codesign.allowed = false

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
