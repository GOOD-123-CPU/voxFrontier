# Security Policy

**English** | [中文](#中文)

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a vulnerability

We take security seriously and appreciate responsible disclosure.

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, use one of these private channels (in order of preference):

1. **GitHub Private Vulnerability Reporting** — open the repository page,
   go to *Security → Report a vulnerability*. This is fully private and
   notifies the maintainers directly.
2. **Email** — send details to the maintainer address listed on the
   repository "About" panel.

Please include as much of the following as you can:

- Type of issue (e.g. data leakage, dependency vulnerability, code injection)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to expect

- **Within 7 days**: acknowledgment of your report.
- **Within 30 days**: an assessment and, if confirmed, a fix or mitigation
  plan with a target release.
- We will credit reporters in the release notes unless you prefer to remain
  anonymous.

## Scope

In scope:

- The Python package `voxfrontier`, its CLI, and the analysis pipeline.
- The synthetic-data generator (e.g. anything that could accidentally emit
  real personal data — it must not, by design).
- The Docker image and CI workflows shipped in this repository.

Out of scope:

- Vulnerabilities in third-party dependencies — please report those to the
  upstream project; we track them via automated dependency updates.
- Issues requiring a compromised local environment.

## Privacy by design

VoxFrontier intentionally ships **only synthetic data**. If you discover any
file, artefact, or git-history object in this repository that appears to
contain real personal data (names, platform identifiers, account numbers,
contact details), treat it as a privacy incident and report it immediately
via the channels above. See `CONTRIBUTING.md` for the local-only privacy
scan we run before every commit.

---

# 中文

## 受支持的版本

| 版本    | 支持情况           |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 报告漏洞

我们非常重视安全问题，并感谢负责任的披露。

**请不要通过公开的 GitHub Issue 报告安全漏洞。**

请优先使用以下私密渠道：

1. **GitHub 私密漏洞报告**：仓库页面 → *Security → Report a vulnerability*，
   全程私密，直接通知维护者。
2. **电子邮件**：发送至仓库 "About" 面板中列出的维护者邮箱。

报告时请尽量包含：问题类型、相关文件路径、受影响代码位置（tag/分支/commit）、
复现步骤、概念验证代码（如有可能）以及影响评估。

### 处理时效

- **7 天内**：确认收到报告。
- **30 天内**：给出评估结论；若确认，提供修复计划与目标版本。
- 除非您要求匿名，我们会在发布说明中致谢报告者。

## 范围

在范围内：`voxfrontier` Python 包、CLI、分析流水线、合成数据生成器
（尤其关注是否可能意外输出真实个人信息——设计上绝不允许）、Docker 镜像与 CI 工作流。

不在范围内：第三方依赖自身的漏洞（请报告给上游项目）、需要已失陷本地环境的场景。

## 隐私即设计

本仓库**只包含合成数据**。若您在本仓库的任何文件、产物或 Git 历史中发现疑似
真实个人信息（姓名、平台标识、账号、联系方式等），请视为隐私事件并立即通过
上述渠道报告。提交前的本地隐私扫描流程见 `CONTRIBUTING.md`。
