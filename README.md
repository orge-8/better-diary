# better-diary

让麦麦每晚写一篇真正像「自己写的」睡前日记。

对比 [bockegai/diary_plugin](https://github.com/bockegai/diary_plugin) 的核心改进：

| 问题（旧插件） | better-diary 的做法 |
|---|---|
| 50k 时间线一次性进 prompt，模型按时间顺序线性复述聊天记录 | **两阶段生成**：阶段一按块提取「值得写的事」并打分，阶段二只喂 top N 精选事件 |
| 「不要写成流水账」和「回忆到现在为止的对话」自相矛盾 | 素材已精选，prompt 只需专注成文；明确「最多写 N 件、没劲就写短」 |
| 日记是转述，没有记忆点 | **强制用「」引用聊天原话**，至少一处 |
| 字数超了靠 `smart_truncate` 硬砍，句子断在半截 | 字数软控制（模型侧遵守 ±80 字），永不硬截断 |
| 天气靠关键词计数猜 | 天气由模型随文生成，贴合当天氛围 |
| AI 腔重（"真是充实的一天"式结尾） | 显式禁令清单：禁总结升华、禁"我意识到"、禁排比、禁连续感叹号 |

## 安装

1. 把 `better-diary/` 整个目录放进 MaiBot 的 `plugins/` 下
2. **升级 / 换版本时，务必先删掉插件目录里的 `config.toml`**（重要，见下方「升级后插件加载失败」）
3. **httpx 是可选依赖**（只有「发布到QQ空间」用得到）：
   - 想要空间发布 → 在主程序虚拟环境里 `pip install httpx`（manifest 已声明该依赖）；
   - 不装也能跑：插件照常加载，日记正常生成并**存档**（`/日记查看` 可看正文），
     仅发布步骤会提示缺少 httpx。此时建议把 `[qzone].enabled` 设为 `false`。
4. 改完 `capabilities` 后需**完整重启 MaiBot**（manifest 在加载前校验，热重载不生效）
5. Runner 会自动生成 `config.toml`，按需修改

> ⚠️ **只替换 `.py` 文件是不够的。** `config.toml` 由 Runner 生成、存放在插件目录内，
> 换版本时它会**原样留在那里**。请整个目录替换，或确认替换后不残留旧 `config.toml`。
> （这正是 better-diary 连续三轮「本地全绿、真机报插件初始化失败」的真正成因。）

**日记成品只发布到 QQ 空间**，聊天里只回报执行状态；想回看正文用 `/日记查看`。

## 命令

| 命令 | 权限 | 说明 |
|---|---|---|
| `/日记` | 管理员 | 生成今天的日记并发布到QQ空间 |
| `/日记 2026-09-26` | 管理员 | 生成指定日期的日记并发布 |
| `/日记查看 [日期]` | 所有人 | 在聊天里回看已存档的日记 |
| `/日记帮助` | 所有人 | 用法说明 |

- 管理员由 `[security].admin_ids` 控制，兼容 `"123456789"` 与 `"qq:123456789"` 写法；**留空时全部放行**；非管理员使用 `/日记` 时**静默忽略**（不回复任何消息，只留日志）。

## 配置

```toml
[plugin]
enabled = true
config_version = "1.0.0"

[diary]
word_target = 250          # 日记正文字数目标
max_events = 3             # 最多写几件事
min_messages = 20          # 当天消息少于此数不生成
filter_mode = "all"        # all / whitelist / blacklist
target_chats = []          # 过滤目标，格式 "group:群号" 或 "private:QQ号"
persona_override = ""      # 留空读取主程序 personality.personality
style_extra = ""           # 追加风格要求，如 "多用叠词"

[schedule]
enabled = true
time = "23:30"             # 每天生成时间（Host 本地时区）
notify_chats = []          # 定时发布后通知哪些聊天，如 ["group:123456"]；留空不发通知

[qzone]
enabled = true             # 日记生成后发布到QQ空间
auto_cookie = true         # 自动获取 cookie：adapter API → NapCat HTTP → 手动兜底，逐级降级
refresh_interval_min = 60  # 自动取 cookie 的节流间隔（分钟）
napcat_http_host = "127.0.0.1"  # NapCat HTTP 服务器地址（1.3.0+ 无 adapter 时的主要来源）
napcat_http_port = "3000"       # NapCat HTTP 服务器端口
napcat_http_token = ""          # NapCat HTTP 服务器 token（未设鉴权留空）
uin = ""                   # 手动兜底 cookie 的 QQ 号（自动获取失败时才用到）
p_skey = ""                # 手动兜底 cookie 的 p_skey（可留空）
skey = ""                  # 手动兜底 cookie 的 skey（可留空）
timeout_seconds = 20       # 发布请求超时（秒）

[llm]
task_name = "utils"        # Host 模型任务名
temperature = 0.8          # 成文温度
extract_temperature = 0.2  # 选材温度（低温保 JSON 稳定）
timeout_seconds = 180      # 单次 LLM 超时（已显式覆盖 RPC 30s 默认值）

[security]
admin_ids = []
```

**关于列表型配置项**：`target_chats` / `notify_chats` / `admin_ids` 标准写法是数组
（`["group:123456"]`），但**写成字符串也能用**——单个值直接写 `"group:123456"`，
多个值用逗号分隔 `"group:111,group:222"`（中英文逗号都认）。v1.2.3 起做了前置归一化，
避免手写格式不对导致整个插件注册失败。

## 工作原理

```
当天消息（get_by_time，按过滤模式）
   │
   ▼
时间线构建（时段标记 + bot 消息标注为"我"）
   │
   ▼  分块（chunk_chars，最多 max_chunks 块，并发选材）
阶段一：每块提取 0-4 件"值得写的事"（who/what/quote/score JSON）
   │  合并 → 按 score 排序 → 去重 → 取 top max_events
   ▼
阶段二：基于精选事件写日记
   首行「2026年9月26日 星期六，晴。」+ 正文（引用原话 + 反 AI 腔规则）
   │
   ▼
存档 data/diaries.json（原子写）→ 发布到QQ空间（说说，纯文本）
```

- **QQ空间发布**：cookie **三级来源逐级降级**——① napcat-adapter API（1.2.x 可用，1.3.0 起 adapter 不兼容自然跳过）→ ② NapCat HTTP 服务器的 `get_cookies`（OneBot 标准动作，与 MaiBot 版本无关，需在 NapCat WebUI 网络配置里开启一个 HTTP Server）→ ③ `[qzone]` 手动填的 cookie 兜底。内存缓存 + 节流 + `data_dir/cookies.json` 原子落盘（权限 0600）；登录态失效（code -3000 等）自动强制重取一次再试。发布接口 `emotion_cgi_publish_v6`，g_tk 用 p_skey 走 5381 算法。
- **发布前语义闸（v1.2.6）**：日记成品是**公开可见**的，所以成文结果必须过一道闸——**模型拒答**（如「抱歉，作为一个人工智能…」）、**空输出**、**残句**（正文不足 20 字）一律判失败，**本日不存档也不发布**，只在日志里留一行 `日记成文不可用（…）`。判定只认强标记，正常日记里的「我一时无法理解他为什么这么说」不会被误伤。
- **外发请求不跟随跳转（v1.2.6）**：发布请求携带登录态 cookie，因此显式 `follow_redirects=False`，并把 cookie 以**域限定**（`user.qzone.qq.com`）写入 client jar。收到 3xx 直接判失败并只回显跳转目标主机名——从结构上杜绝「跳转把凭据带去别的域」。
- **选材降级**：阶段一失败或选出 0 件时，自动降级用当天时间线末尾 4000 字作素材，流程不中断。
- **RPC 超时**：LLM 调用显式带 `timeout_ms`，绕开 `cap.call` 默认 30 秒限制。
- **存档**：`ctx.paths.data_dir/diaries.json`，临时文件 + `replace` 原子写；读取坏文件当空处理。
- **日志脱敏（v1.2.6）**：cookie 相关异常与 adapter 返回的 error 字段在写日志前统一脱敏（`p_skey=<redacted>`），避免登录态被写进日志文件。
- **昵称净化（v1.2.6）**：用户昵称是不可信输入，拼进 prompt 前去掉花括号/方括号/换行，避免被拿来伪造时间线结构。

## 故障排查

**Q: 日记生成到一半报「LLM 调用失败」？**
看日志里的具体 error。超时可调大 `[llm].timeout_seconds`；确认 `[llm].task_name` 是 Host 模型任务页里存在的任务名。

**Q: 定时任务没跑？**
确认 `[schedule].enabled = true` 且 MaiBot 在设定时间处于运行状态（调度器随插件加载启动，错过当天不补跑）。

**Q: 发布失败，提示登录态失效 / 重取失败？**
`auto_cookie = true` 时插件会自动重取 cookie 再试一次。MaiBot 1.2.x 走 napcat-adapter；**1.3.0 起 adapter 不兼容，请在 NapCat WebUI → 网络配置里开一个 HTTP 服务器**（记下端口和 token，填进 `[qzone].napcat_http_*`），插件会直连 NapCat 拿 cookie，与 MaiBot 版本无关。都没有时才需要手动填 `[qzone]` cookie。

**Q: 提示「拿不到QQ空间cookie」？**
三级来源都没取到：① adapter 未装载/不可用；② NapCat HTTP 未开启或 host/port/token 不对（403 = token 错）；③ 手动兜底没填。按 ①→②→③ 排查，推荐直接用 ②。

**Q: 定时发布后会在群里说话吗？**
默认**不会**（完全静默，只写日志）。想让某个群/私聊收到通知，把 `[schedule].notify_chats = ["group:群号"]` 填上，发布成功/失败都会收到一条短消息。

**Q: 想只给某个群写日记？**
`[diary].filter_mode = "whitelist"`，`target_chats = ["group:群号"]`。

**Q: 定时到点了，群里也没动静、空间里也没发新日记？**
先看日志有没有 `日记成文不可用（…）`。v1.2.6 起，**模型拒答 / 空输出 / 正文不足 20 字**会被发布前语义闸拦下——
本日既不入档也不发布（宁可空一天，也不把「抱歉，作为一个人工智能…」发到公开空间）。常见诱因：
`[llm].task_name` 指到了能力弱或不支持长输出的模型、`[llm].timeout_seconds` 太短导致半截返回。
日志里同时会打出模型原文前 80 字，据此判断是拒答还是截断。

**Q: 日记味道不对？**
先调 `style_extra`（会追加进写作规则）；还不行就改 `persona_override` 完整接管人设描述。

**Q: 启动就失败，报 `No module named 'bd_prompts'`？**
v1.2.1 已修。根因是 Runner **把插件目录当包加载**（目录不在 `sys.path` 上），此时只有相对导入
`from .bd_prompts import ...` 能解析；旧代码用的是平铺导入 `from bd_prompts import ...`，真机必然挂。
修法是双路径导入（相对优先、平铺兜底）+ 补 `__init__.py` 包标记。若你手上是旧版，整目录换 v1.2.1 即可。

> 注意：这类问题**本地冒烟测不出来**——旧的冒烟测试用平铺方式加载，跟真机加载方式不同，
> 所以出现过「门禁全绿但真机启动失败」。v1.2.1 起冒烟测试改为包式加载（显式传
> `submodule_search_locations` 并保证插件目录不在 `sys.path`），与真机一致。

**Q: 报「插件初始化失败」/ 仍提示 `No module named 'bd_prompts'`？**
v1.2.2 已修。**报错信息具有误导性**：真实原因是真机 MaiBot 的 Python 环境**没装 httpx**，
而 `bd_cookie.py` / `bd_qzone.py` 曾在模块级裸 `import httpx`；`plugin.py` 顶层导入它时抛
`ImportError`，被兜底分支误判成「包式加载不可用」，于是回退平铺导入并报出 `bd_prompts` 找不到——
**真因被掩盖了**。

v1.2.2 两处改动根治：

- httpx 改为**可选依赖**（模块内 `try: import httpx` + `HTTPX_AVAILABLE` 标志）。
  没有 httpx 时插件**照常加载**，只有「发布到QQ空间」这一步不可用并给出可读提示；
  日记生成与存档不受影响。
- 导入分支改用 `if __package__:` 精确判断，不再用宽泛的 `except ImportError`
  吞掉下游依赖错误——以后任何依赖问题都会报出真名。

想彻底用上空间发布，二选一：① 在 MaiBot 的 Python 环境里装 httpx（`pip install httpx`）；
② 暂时 `[qzone].enabled = false`，日记仍会正常生成并存档，用 `/日记查看` 取正文。

**Q: 报「插件注册失败 / 插件初始化失败」，且日志里没有本插件任何输出？**

**先试这一步：删掉插件目录里的 `config.toml`，然后完整重启 MaiBot。**

这是本插件真机上**唯一实际发生过**的失败原因，已实测确认。典型场景是
**MaiBot 升级**：`plugin.config_version` 是 **Host 1.2.3 才引入的硬性要求**，而插件目录里那份
`config.toml` 是**旧宿主生成**的、没有这个键。升级后 Runner 一读就判非法：

```text
插件配置版本非法: 插件配置文件缺少 plugin.config_version 版本号，当前版本策略不再兼容无版本配置
```

删掉它，Runner 会按当前配置模型重新生成一份合法配置（只有 `[plugin]` 与你改过的项会重置，
**存档的日记不会丢**——日记在 `data/plugins/` 里，不在插件目录）。
另一种等价做法：在 WebUI 里保存一次该插件的配置。

**为什么日志一片空白？** 因为这个检查发生在 Runner（`runner/runner_main.py::extract_plugin_config_version`）
里，位置在**「配置注入」**阶段——比 `on_load` 更早，此时插件代码一行都还没执行：

```
扫描 plugins/ → 读 _manifest.json（严格校验）→ 校验依赖
   → 导入 plugin.py → create_plugin() → 注入 ctx
   → 配置注入：读 config.toml → 版本检查 ✗   ← 就死在这里
   → on_load()                              ← 这层的失败才报「插件加载失败」
```

> 判据：**「插件初始化失败」≠「插件加载失败」。** 后者才是 `on_load` 抛的异常。
> 看到「初始化失败」就说明还没进你的代码——**不要再去改插件源码**。

**这也意味着插件侧代码无法自救**：拦截点全在宿主，插件能做的只有「换个合法的 config.toml」。
v1.2.2 / v1.2.3 / v1.2.4 做的配置容错（`field_validator` 归一化、覆写 `set_plugin_config`、
逐字段修复）**都不是针对这条路的**——它们仍有价值（覆盖「配置能过版本检查、但字段类型写错」
那类问题），但**治不了上面的场景**。真正有效的动作只有「删 `config.toml` 重生成」。

配置格式参考（Runner 会自动生成，正常不必手写）：

```toml
[plugin]
enabled = true
config_version = "1.0.0"       # ✓ 必须有

[security]
admin_ids = ["123456789"]      # ✓ 标准写法；写成 "123456789" 也接受（v1.2.3 起）
```

若确认不是上面这条（即日志里**有**本插件的输出），再去搜这几句：

| 日志关键字 | 含义 |
|---|---|
| `日记成文不可用（…）` | 模型拒答 / 空输出 / 残句被语义闸拦下，本日不存档、不发布（后面跟着模型原文前 80 字） |
| `插件配置注入失败，已回退…` | 配置能过版本检查、但字段类型有问题，后面跟着真实异常 + traceback |
| `回退配置无法逐字段修复…` | 连逐字段修复都失败，本次用模型默认值运行 |
| `插件配置不可用，已回退默认配置继续加载…` | 配置实例缺失（下游保险生效） |

> 提醒：配置类问题**只有走真实 SDK 的 `set_plugin_config` 才测得出来**。旧冒烟测试的
> `[0c]` 直接调 `BetterDiaryConfig.model_validate`，绕过了版本检查，所以它永远是绿的。
> v1.2.4 新增 `[0d]`：把 12 种坏配置灌进真实调用链，断言零异常。

