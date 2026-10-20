# UEパッケージによる動画生成

リポジトリのルートから、推論環境のPythonで実行します。

```bash
python ue/run_video.py --package /path/to/Normal --video /path/to/input.mp4
```

コード・モデル・標準の出力先は、このリポジトリの場所から解決します。推論とカメラ推定には、起動したPythonを使用します。ユーザー名、condaの配置場所、WSLの名前を設定する必要はありません。

## 環境とモデルの準備

推論環境の作成とモデルの取得・配置は、ルートREADMEの[環境の準備](../README.md#環境の準備)と[モデルの取得と配置](../README.md#モデルの取得と配置)に従ってください。

## UEパッケージ

UEパッケージは、GitHub Releasesで配布しているUEプロジェクトから作成します。SMPLモデルは再配布できないため、UEプロジェクトには含めていません。推論用に配置した `body_models/smpl/SMPL_FEMALE.pkl` から人体を作り、次の手順でプロジェクトへ配置してからパッケージを作成してください。

### 必要なもの

- Windows 10/11、Unreal Engine 5.5.4、Visual Studio 2022（C++によるゲーム開発ワークロード）
- UEプラグイン Simple TCP UDP Socket Client（作者 Socke、プラグイン名 `SocketClient`。[Fab](https://www.fab.com/)で取得し、UE 5.5のエンジンにインストール）
- ルートREADMEの手順で作成した `hmr` 環境と、配置済みの `body_models/smpl/SMPL_FEMALE.pkl`

### SMPLの配置とパッケージ作成

1. [Releases](https://github.com/gyantic/research_repo_hmr/releases/tag/ue-project-v1) から `ue-unreal-project.zip` を取得し、Windows上の任意のフォルダに展開します。展開すると `unreal` フォルダができます。

   ```powershell
   Expand-Archive ue-unreal-project.zip -DestinationPath C:\path\to\folder
   cd C:\path\to\folder\unreal
   ```

   以降は、この `unreal` フォルダをカレントディレクトリとしたWindows PowerShellのコマンドです。UEエディタは閉じた状態で実行します。

2. エディタ用モジュールをビルドします。SMPLのインポートには、このモジュールを読み込めるUEエディタが必要です。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\build-editor.ps1
   ```

3. `SMPL_FEMALE.pkl` から人体のglTFを作成し、`unreal` フォルダの `Interchange/SMPL.glb` に書き出します。`hmr` 環境で、このリポジトリのルートから実行します。WSLから実行する場合、`unreal` フォルダは `/mnt/c/path/to/folder/unreal` のように指定します。

   ```bash
   conda activate hmr
   python /path/to/folder/unreal/Tools/SMPLSetup/smpl_to_gltf.py body_models/smpl/SMPL_FEMALE.pkl /path/to/folder/unreal/Interchange/SMPL.glb
   ```

4. glTFを `/Game/SMPL`（スケルトン `/Game/SMPL_Skeleton`）としてインポートし、受信Blueprintの人体に設定します。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\import-smpl.ps1
   ```

5. Normal・RockDrop・SphereOrbitの3パッケージを作成します。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\package-recording.ps1
   ```

   パッケージは `Deliverables/{Normal,RockDrop,SphereOrbit}` に作成されます。3つは同じマップを使用し、`package-manifest.json` の演出（`scene_mode`）だけが異なります。

生成される `Interchange/SMPL.glb`、`Content/SMPL.uasset`、`Content/SMPL_Skeleton.uasset` はSMPLを、`Deliverables/` はSMPLとSocketClientプラグインを含むため、再配布しないでください。

衣装は人体とは別のスケルタルメッシュ（`/Game/Garments/SK_Garments`）です。録画開始時に `ClothMeshCompm` を人体 `SMPLcomp` の姿勢に追従させます。

### 同梱素材のクレジット

- シャツ：["T-shirt Wankil brodé"](https://sketchfab.com/3d-models/t-shirt-wankil-brode-3232348fabd144efb76aa2dda53d8f7f) by Angerenage, licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- ズボン：["cloths"](https://sketchfab.com/3d-models/cloths-815ccf1a7fc046b2a3ca9ca8c7b2983f) by salman_ali, licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

衣装はSMPLのスケルトンに合わせてスキニングし、人体部分を除去しています。

### パッケージの指定

Normal・RockDrop・SphereOrbitのいずれかを指定します。`--package` に指定するフォルダは次の構成です。

```text
<package>/
  package-manifest.json
  Windows/
    MyProject_ROMP.exe
    ...（パッケージのほかの同梱ファイル）
```

現在のUEパッケージはWin64用です。Windowsでは直接起動し、WSLから実行する場合はWindows相互運用を使ってUEを起動します。UEへ渡すファイルパスだけをWindows形式へ変換します。推論や合成は起動元の環境で動きます。

UE録画にはWindows版FFmpegも使用します。通常はWindowsのPATHから検出します。別の実行ファイルを使用する場合は `--recorder-ffmpeg` で指定します。推論側の `--ffmpeg`・`--ffprobe` は起動元の環境で使用する実行ファイルです。

## 実行と出力

```bash
conda activate hmr
python ue/run_video.py --package /path/to/Normal --video /path/to/input.mp4
```

`package-manifest.json` に従って演出を選びます。入力動画ごとにパッケージを作り直す必要はありません。パッケージと動画のパスには、起動元のOSで参照できるパスを指定してください。

標準の出力先は次のとおりです。

```text
results/ue/<Package>/Recordings/<timestamp>/composite/final.mp4
```

`<Package>` は `package-manifest.json` の `scene_mode` から決まり、Normal・RockDrop・SphereOrbitのいずれかになります。展開したフォルダ名は使いません。

任意の出力先は `--output` で指定します。既存のディレクトリは受け付けません。

カメラ推定だけ別のPython環境を使用したい場合は、任意で `--observation-python` を指定できます。通常は指定不要です。その他のオプションは `python ue/run_video.py --help` を参照してください。
