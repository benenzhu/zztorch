# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pip install googletrans

from googletrans import Translator
trans = Translator(service_urls=['translate.google.cn'])

trans.translate('The Quick Brown fox')

trans = Translator()
print(trans.translate('星期天','en'))

# +
from googletrans import Translator

# 实例化翻译器，由于模块默认的服务url在国内无法使用，所以我们修改成国内可用的google翻译服务地址
translator = Translator(service_urls=["translate.google.cn"])

# 调用翻译函数，指定原语言的代码(en)，和目标语言的代码(zh-CN)
result = translator.translate('Hello, I am Big Tree!', src='en', dest='zh-CN')

# 原语言代码
print(result.src)
# 目标语言代码
print(result.dest)
# 要翻译的内容
print(result.origin)
# 翻译后的内容
print(result.text)
# 翻译后内容的发音
print(result.pronunciation)

# -


