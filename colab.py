import time
import datetime
import webbrowser

# 1時間毎に任意のノートブックを開く
for i in range(12):
    browse = webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
    browse.open('https://colab.research.google.com/drive/1VDW2N9ipMkeEQsAOJ1lABrX8i-QI1KuP#scrollTo=lckGpHvMC-p5')
    print(i, datetime.datetime.today())
    time.sleep(60*60)