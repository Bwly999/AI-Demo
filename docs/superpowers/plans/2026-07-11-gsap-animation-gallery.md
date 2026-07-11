# GSAP Animation Gallery Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建包含十个可重播场景的响应式 GSAP 单页动画画廊。

**Architecture:** 使用单一 HTML 定义语义结构，CSS 负责独立场景视觉，JavaScript 通过场景注册表创建和管理 GSAP 时间线。所有动画限定在卡片作用域内，并统一支持重播和减少动态效果。

**Tech Stack:** HTML5、CSS3、JavaScript、GSAP 3.13、MotionPathPlugin

---

### Task 1: 页面结构与视觉系统
- [ ] 创建 `gsap-animation-gallery/index.html`
- [ ] 创建 `gsap-animation-gallery/styles.css`
- [ ] 完成响应式卡片、焦点状态和场景静态图形

### Task 2: 十个动画场景
- [ ] 创建 `gsap-animation-gallery/app.js`
- [ ] 注册 MotionPathPlugin
- [ ] 实现十个作用域动画工厂
- [ ] 添加单卡重播和全部重播
- [ ] 支持 prefers-reduced-motion

### Task 3: 浏览器验证
- [ ] 启动本地静态服务器
- [ ] 检查桌面与移动布局
- [ ] 检查控制台错误和交互
- [ ] 保存最终截图
