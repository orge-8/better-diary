# -*- coding: utf-8 -*-
"""better-diary 插件包。

Runner 既可能把插件目录当包加载（相对导入可用），也可能只把 plugin.py 当顶层
模块加载（需要目录在 sys.path 上）。plugin.py 中的自建模块统一用双路径导入
兼容两种情形：

    try:
        from .bd_prompts import ...      # 包式加载（Runner 真机）
    except ImportError:
        from bd_prompts import ...       # 平铺兜底（脚本直跑 / 旧测试）

v1.2.1 起补上本文件：此前 plugin.py 只有平铺导入、又假设插件目录在 sys.path 上，
真机包式加载时直接 `No module named 'bd_prompts'` 启动失败。
"""
