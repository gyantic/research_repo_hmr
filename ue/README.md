# ue（UEパッケージによる演出動画の生成）

本ディレクトリには、入力動画から人物姿勢とカメラパラメータを推定し、Unreal Engine（UE）パッケージによるレンダリング・録画を経て、実写映像と合成した最終動画を出力するパイプライン（`run_video.py`）が含まれています。

研究全体の構成については[ルートREADME](../README.md)を参照してください。

---

## クイックスタート

環境構築・モデル配置・UEパッケージの作成が完了していれば、リポジトリのルートから推論環境（`hmr`）のPythonで以下のコマンドを実行するだけで合成動画を生成できます。

```bash
conda activate hmr
python ue/run_video.py --package /path/to/Normal --video /path/to/input.mp4
```

- コード・モデル・標準の出力先パスは、すべてこのリポジトリのルート位置から自動的に解決されます。
- 姿勢推論とカメラ推定には、このスクリプトを起動したPython環境がそのまま使用されます。OSのユーザー名、Condaのインストール先、WSLのディストリビューション名などを個別に設定する必要はありません。

---

## 1. 環境とモデルの準備

推論用Conda環境（`hmr`）の作成およびモデル重み（SMPL・SAT-HMR・MoGe-2）の取得・配置については、ルートREADMEの以下のセクションに従ってください。

- [1. 環境の準備](../README.md#環境の準備)（`1.2 UE動画生成用の環境（hmr）` および `1.3 chumpyの互換修正`）
- [2. モデルの取得と配置](../README.md#モデルの取得と配置)（`SMPL_NEUTRAL.pkl`、`SMPL_FEMALE.pkl`、SAT-HMR補助データ、`sat_644.pth`、MoGe-2 `model.pt`）

---

## 2. UEパッケージの作成

演出用のUEパッケージは、GitHub Releasesで配布しているUEプロジェクトからビルドします。SMPLモデルはライセンス上再配布できないため、配布プロジェクトには含めていません。推論用に配置した `body_models/smpl/SMPL_FEMALE.pkl` から人体メッシュを生成し、以下の手順でプロジェクトへインポートしてからパッケージをビルドしてください。

### 必要なもの

- Windows 10 / 11、Unreal Engine 5.5.4、Visual Studio 2022（「C++によるゲーム開発」ワークロード）
- UEプラグイン **Simple TCP UDP Socket Client**（作者：Socke、プラグイン名：`SocketClient`）：[Fab](https://www.fab.com/listings/d19071be-473f-4dcb-8a95-ced3eda6667d) で取得し、Epic Games Launcherから UE 5.5 のエンジンにインストールしてください
- ルートREADMEの手順で作成した `hmr` 環境と、配置済みの `body_models/smpl/SMPL_FEMALE.pkl`

### SMPLの配置とパッケージビルド手順

1. [GitHub Releases](https://github.com/gyantic/research_repo_hmr/releases/tag/ue-project-v1) から `ue-unreal-project.zip` を取得し、Windows上の任意のフォルダに展開します。展開すると `unreal` フォルダが生成されます。

   ```powershell
   Expand-Archive ue-unreal-project.zip -DestinationPath C:\path\to\folder
   cd C:\path\to\folder\unreal
   ```

   以降の手順2・4・5は、この `unreal` フォルダをカレントディレクトリとした**Windows PowerShell**のコマンドです。UEエディタは閉じた状態で実行してください。

2. エディタ用モジュールをビルドします（SMPLのインポートには、このモジュールを読み込めるUEエディタが必要です）。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\build-editor.ps1
   ```

3. `SMPL_FEMALE.pkl` から人体のglTFを作成し、`unreal` フォルダ内の `Interchange/SMPL.glb` に書き出します。**`hmr` 環境を有効化し、このリポジトリのルートから**実行します（WSLから実行する場合、`unreal` フォルダは `/mnt/c/path/to/folder/unreal` のように指定します）。

   ```bash
   conda activate hmr
   python /path/to/folder/unreal/Tools/SMPLSetup/smpl_to_gltf.py body_models/smpl/SMPL_FEMALE.pkl /path/to/folder/unreal/Interchange/SMPL.glb
   ```

4. 生成したglTFを `/Game/SMPL`（スケルトン `/Game/SMPL_Skeleton`）としてインポートし、受信Blueprintの人体に設定します（Windows PowerShellで `unreal` フォルダから実行）。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\import-smpl.ps1
   ```

5. `Normal`・`RockDrop`・`SphereOrbit` の3パッケージをビルドします（Windows PowerShellで `unreal` フォルダから実行）。

   ```powershell
   powershell -ExecutionPolicy Bypass -File Tools\package-recording.ps1
   ```

   パッケージは `Deliverables/{Normal,RockDrop,SphereOrbit}` に作成されます。3つのパッケージは同じマップを使用しており、`package-manifest.json` 内の演出設定（`scene_mode`）のみが異なります。

> **Warning:** 生成される `Interchange/SMPL.glb`、`Content/SMPL.uasset`、`Content/SMPL_Skeleton.uasset` にはSMPLデータが、また `Deliverables/` 配下のパッケージにはSMPLデータとSocketClientプラグインが含まれるため、第三者へ再配布しないでください。

### 衣装アセットの仕組みとクレジット

衣装は人体とは別のスケルタルメッシュ（`/Game/Garments/SK_Garments`）として構成されており、録画開始時に衣装コンポーネント（`ClothMeshCompm`）を人体コンポーネント（`SMPLcomp`）の姿勢に追従させています。衣装メッシュはSMPLのスケルトンに合わせてスキニングし、内部の人体部分を除去して使用しています。

<a id="同梱素材のクレジット"></a>

#### 同梱素材のクレジット

- **シャツ**：["T-shirt Wankil brodé"](https://sketchfab.com/3d-models/t-shirt-wankil-brode-3232348fabd144efb76aa2dda53d8f7f) by Angerenage, licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- **ズボン**：["cloths"](https://sketchfab.com/3d-models/cloths-815ccf1a7fc046b2a3ca9ca8c7b2983f) by salman_ali, licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

---

## 3. 実行と出力

### パッケージの指定と動作環境（Windows / WSL）

`--package` には、ビルドした `Normal`・`RockDrop`・`SphereOrbit` のいずれかのフォルダを指定します。各パッケージフォルダは以下の構成になっています。

```text
<package>/
  package-manifest.json
  Windows/
    MyProject_ROMP.exe
    ...（パッケージのほかの同梱ファイル）
```

- **Windows / WSLの相互運用**：現在のUEパッケージはWin64用です。Windows環境から実行した場合は直接起動し、WSL環境から実行した場合はWindows相互運用機能を介してUE（`.exe`）を起動します。この際、UEへ渡すファイルパスのみが自動的にWindows形式へ変換され、姿勢推論や動画合成は起動元の環境で実行されます。
- **FFmpegの使い分け**：UEの録画処理にはWindows版のFFmpegも使用します（通常はWindows側の `PATH` から自動検出されますが、別の実行ファイルを指定したい場合は `--recorder-ffmpeg` を使用します）。一方、推論・合成側の `--ffmpeg`・`--ffprobe` オプションには、起動元のOS環境で使用する実行ファイルを指定します。

### コマンドの実行と出力先

```bash
conda activate hmr
python ue/run_video.py --package /path/to/Normal --video /path/to/input.mp4
```

- パッケージ内の `package-manifest.json` に従って演出が選択されるため、入力動画ごとにパッケージを作り直す必要はありません。
- `--package` と `--video` には、コマンドを実行する起動元のOSから参照できるパスを指定してください。

標準の出力先は以下のとおりです。

```text
results/ue/<Package>/Recordings/<timestamp>/composite/final.mp4
```

- `<Package>` の名前は展開したフォルダ名ではなく、`package-manifest.json` の `scene_mode` から決定されます（`Normal`・`RockDrop`・`SphereOrbit` のいずれか）。
- **出力先の変更**：任意の出力先を指定する場合は `--output` を使用します（上書き防止のため、既に存在するディレクトリは指定できません）。
- **その他のオプション**：カメラ推定のみ別のPython環境を使用したい場合は `--observation-python` で指定可能です（通常は指定不要）。全オプションの一覧は `python ue/run_video.py --help` で確認してください。
