# 用 GitHub Actions + NuGet Trusted Publishing 发布

`publish-nuget.yml` 不读取任何长期 NuGet API Key：
GitHub 的 OIDC token → nuget.org 校验 trusted publishing policy →
`NuGet/login@v1` 换 1 小时有效的临时 API key → `dotnet nuget push`。
下面是一次性配置。

## 0. 准备

- 仓库：`https://github.com/dongfangzhizhu/H3NetRuntime`（默认分支 `main`）。
- nuget.org 账号：`wenlee`，包 ID `H3NetRuntime`（发布前已确认该 ID 未被占用）。
- 仓库里 `dist/win-x64/h3rt.dll`、`dist/linux-x64/libh3rt.so`、
  `assets/tokenizer/tokenizer.json` 必须存在（CI 的打包步骤会校验）。
  它们按 `sm_89` 编译；给别的算力发布前先本地重建并替换。

## 1. nuget.org Trusted Publishing policy（已配置）

| 字段 | 值 |
|---|---|
| Package owner | `wenlee` |
| Scopes | Push new packages and package versions |
| Glob Patterns and Packages | `*` |
| Publisher | GitHub Actions |
| Repository Owner | `dongfangzhizhu` |
| Repository | `H3NetRuntime` |
| Workflow | `publish-nuget.yml`（只填文件名） |
| Environment | `production` |

该策略目前是**临时激活，7 天内必须成功发布一次**才会永久生效，因此尽快走完第 3 节。
工作流里的 `environment: production` 与 `user: wenlee` 已按此策略写死
（`NUGET_USER` 仓库变量可覆盖用户名，非必填）。

## 2. 在 GitHub 配置仓库

1. **Settings → Environments**：确保存在名为 `production` 的 environment，
   必须与 nuget.org 策略里的 Environment 一致。可添加 required reviewers
   作为发布前的审批关卡；若审批开启，发布会先暂停等待批准。
2. （可选）**Settings → Secrets and variables → Actions → Variables**：
   `NUGET_USER = wenlee`。不填时工作流默认使用 `wenlee`。
3. 确认 Actions 有权限写 OIDC token：工作流里已声明
   `permissions: id-token: write`，无需额外设置。

## 3. 发布

方式 A —— 打 tag（推荐）：

```bash
git tag v1.0.0
git push origin v1.0.0
```

tag 去掉前导 `v` 就是包版本（`v1.0.0` → `1.0.0`）。

方式 B —— Actions → **Publish NuGet** → **Run workflow**，手动输入版本号。

工作流会还原、构建、打包，校验包内两个原生库与 tokenizer，
然后推送到 https://api.nuget.org/v3/index.json。
`--skip-duplicate` 让重复运行同一版本安全失败（不会覆盖已有包）。

## 4. 验证

- Actions 页面出现绿色 run，日志末尾是 `Successfully pushed .nupkg`。
- https://www.nuget.org/packages/H3NetRuntime 显示新版本（索引需要几分钟）。
- 本地验证：

  ```bash
  dotnet add package H3NetRuntime --version 1.0.0
  ```

## 5. 排错

| 现象 | 处理 |
|---|---|
| `The response status code did not indicate success` / 401 | 用户名不是 `wenlee`（`NUGET_USER` 必须是 nuget.org profile name，不是邮箱）；或策略未生效 |
| 403 / `no matching trusted publishing policy` | 策略的 owner / repo / workflow 文件名 / environment 与工作流不一致；当前应为 `dongfangzhizhu` / `H3NetRuntime` / `publish-nuget.yml` / `production` |
| 策略显示 “Use within 7 days” | 这是临时激活状态；当前策略必须在 7 天内成功发布一次，之后自动永久生效 |
| `Package ID already exists` / 无法推送 | 你的账号不拥有该包 ID；改 `PackageId` 或在 nuget.org 申请所有权 |
| `package version already exists` | NuGet 不允许覆盖；递增版本，重打 tag |
| 包里没有原生库 | `dist/` 下的文件未提交，或被 `.gitignore` 排除；工作流会提前失败 |
| GPU 算力不匹配 | 预编译包是 `sm_89`；用 `scripts/build.cmd <算力>` / `scripts/build.sh <算力>` 重建后提交，再发新版本 |

本地不打 tag 也能打包：

```bat
scripts\pack.cmd 89
```

产物在 `dist/nuget/`，它只是本地文件，不会自动发布。
