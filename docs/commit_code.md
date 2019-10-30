## 開發流程

experiment這個branch是大家工作的統合成果，每個人的程式碼最後都會merge到experiment這個branch, 然在開發過程可能會做多次改動，並非一次commit就可以交付成果，若中介commit直接上到experiment branch, 會讓experiment出現build fail或甚至是無法執行等問題，為了兼顧開發上的便利性以及experiment branch的穩定性，我們在開發上使用以下的流程。

為了說明上的方便，假設您要新加一個功能進來，流程會是

```
# 從最新的experiment切出來
$ git checkout experiment
$ git pull  # 更新到最新的code
$ git checkout -b awesome_feature  # 新增並切到 awesome_feature branch。

# 重覆以下兩動直到feature完成：
$ 改code
$ git add your_code.cpp
$ git commit

# OK, 新功能已加到awesome_feature branch裡，要push回gitlab了
$ git push
```

接下來就是送出merge request:
1. 到[gitlab](https://gitlab.itriadv.co/self_driving_bus/self_driving_bus)的頁面。
2. 點選左側欄的Merge Requests。
3. 點選New Merge Request綠色按鈕。
4. 在Select source branch點一下並選擇awesome_feature，右方的target branch選experiment，然後點下Compare branches and continue綠色按鈕。
5. 填入以下資訊，其中Title和Description類推適用git commit時的動作。
    * Title
    * Description
    * 點選Assignee，挑選一個適當的reviewer
    * **不要**打勾 Squash commits when merge request is accepted
6. 點下Submit merge request.

系統會自動寄信給Reviewer, 當他Review完及點下approve後, code就會自動merge到experiment branch裡，整個流程就告一段落。
