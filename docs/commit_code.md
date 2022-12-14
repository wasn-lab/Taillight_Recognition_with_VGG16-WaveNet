### 開發流程

大家工作的統合成果放在 master branch，
然在開發過程可能會做多次改動，並非一次commit就可以交付成果，若中介commit直接上到master
branch, 會讓master出現 build
fail或甚至是無法執行等問題，為了兼顧開發上的便利性以及master
branch的穩定性，我們在開發上使用以下的流程。

為了說明上的方便，假設您要新加一個功能進來，流程是先從master
branch分支出來一個自己在用的branch (姑且稱為awesome_feature)，
修改回之後push 到repository，
新增一個merge request，
reviewer 完之後，就會自動merge回master branch裡，詳細的指令如下。

```
# 下載repository到local電腦
$ git clone https://gitlab.itriadv.co/self_driving_bus/itriadv.git
$ git branch
* master   # We are at branch master
$ git checkout -b awesome_feature  # 新增並切到 awesome_feature branch。

# 重覆以下兩動直到feature完成：
$ 改code
$ git add your_code.cpp
$ git commit

# OK, 新功能已加到awesome_feature branch裡，要push回gitlab了
$ git push
```

接下來就是送出merge request:
1. 到 gitlab的 [itriadv 頁面](https://gitlab.itriadv.co/self_driving_bus/itriadv)。
2. 點選左側欄的Merge Requests。

![init_nav.png](images/init_nav.png)

3. 點選New Merge Request綠色按鈕。
4. 在Select source branch點一下並選擇awesome_feature，右方的target branch選master，然後點下Compare branches and continue綠色按鈕。

![new_mr.png](images/new_mr.png)

5. 填入以下資訊，其中Title和Description類推適用git commit時的動作。
    * Title
    * Description
    * 點選Assignee，挑選一個適當的reviewer
    * **不要**打勾 Squash commits when merge request is accepted
6. 點下Submit merge request.

![new_mr.png](images/submit_mr.png)


系統會自動寄信給Reviewer, 當他Review完及點下approve後, code就會自動merge到master branch裡，整個流程就告一段落。

### 挑選 Reviewer

Reviewer需要有maintainer權限，目前有
@Austin.Chen
@Wayne
@chtseng
@hankliu
@BensonHuang
@chinghao.liu

### Review 注意事項

為了確定master branch處於可以build pass並能順利執行，Review時請注意以下事項。

- 在 Review 頁面，下方有個 Changes 的 tab, 點下去可以看到改了什麼，要看程式是否有考慮不週之處。
- 是否有誤上檔案，如 log 檔
- 點下 Merge button之前，看一下上方pipeline的執行結果，pipeline執行失敗的話，強行merge必然會出現build fail，不可點下Merge。

### 常見問題

#### 要怎麼把master branch裡的code merge到自己的local branch?

以下為例，我們想要把最新的master branch merge 回camera_grabber裡

```
# 檢查目前local repository的設定
$ git branch -a
* camera_grabber  # 目前所在的branch
  master          # local repository裡的master branch
  remotes/origin/camera_grabber
  remotes/origin/control_team
  remotes/origin/docs
  remotes/origin/drivenet
  remotes/origin/fusion_d
  remotes/origin/localization
  remotes/origin/master  # remote repository裡的master branch，這是我們要merge的來源
  remotes/origin/parknet
  remotes/origin/tracking_pp

# 把最新的code抓回來
$ git fetch
remote: Enumerating objects: 62, done.
remote: Counting objects: 100% (62/62), done.
remote: Compressing objects: 100% (47/47), done.
remote: Total 62 (delta 3), reused 0 (delta 0)
Unpacking objects: 100% (62/62), done.
From ssh://gitlab.itriadv.co:7022/self_driving_bus/itriadv
   80b4a3b..ca84b9f  control_team -> origin/control_team
   54159f0..cd455d7  master     -> origin/master

# 把master merge到camera_grabber裡
$ git merge remotes/origin/master --no-ff
Merge made by the 'recursive' strategy.
 src/control/GUI_Publisher/CMakeLists.txt                                         |    67 +
...
```

注意，git merge 的來源是remotes/origin/master, 不是自己local裡的master喔。


#### merge request pipeline 回報failure, 改完code之後要再新建一個merge request嗎?

不用再新建merge request。 merge request是把branch最新的code合進master裡，所以只要把修改完的code push上來，就會重啟pipeline的程序並更新其狀態。

#### 要如何啟動merge request的builder做clean build?

為了快速執行完pipeline, builder會在package.xml有改動時才會做clean build,
故只要任一個package.xml檔有改動（如加一個空白行），然後commit上去，builder就會做clean build。

#### 要如何把車上修改過的程式傳回桌機並commit回repository?

在特殊情況（如車控）非得要在車上改code並驗証，若要將修改的code傳回來並commit回repo，方法不只一個，
這裡只列一種：
```
# 在車上的電腦把修改存成.patch檔
$ git diff > my_patch.patch
$ scp my_patch.patch chtseng@ci.itriadv.co:/tmp  # 把.patch傳回到桌機的/tmp下

# 在桌機：
$ cd /path/itriadv  # 一定要切回itriadv repo的根目錄
$ git apply --check /tmp/my_patch.patch  # (Optional) 若程式沒有錯誤訊息則執行下一指令
$ git apply /tmp/my_patch.patch
$ git status # 檢查apply後的結果
```
