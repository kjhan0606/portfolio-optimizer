"""
리스크 최소화 주식 포트폴리오 최적화 시스템
Android App (Kivy/KivyMD)
"""

import os
import threading
import tempfile
from datetime import datetime, timedelta

from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image as KivyImage
from kivy.uix.widget import Widget
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.list import OneLineIconListItem, IconLeftWidget
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.spinner import MDSpinner

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ══════════════════════════════════════════
# Constants
# ══════════════════════════════════════════
POPULAR_STOCKS = {
    '삼성전자': '005930', 'SK하이닉스': '000660', 'LG에너지솔루션': '373220',
    '삼성바이오로직스': '207940', '현대차': '005380', 'NAVER': '035420',
    '카카오': '035720', 'LG화학': '051910', 'POSCO홀딩스': '005490',
    '삼성SDI': '006400', '기아': '000270', 'KB금융': '105560',
    '셀트리온': '068270', '신한지주': '055550', '하나금융지주': '086790',
    '현대모비스': '012330', 'LG전자': '066570', 'SK이노베이션': '096770',
    '한국전력': '015760', 'KT&G': '033780', '삼성물산': '028260',
    '삼성생명': '032830', 'HMM': '011200', '두산에너빌리티': '034020',
    'SK텔레콤': '017670', '한화에어로스페이스': '012450', 'LG': '003550',
}

POPULAR_ETFS = {
    'KODEX 200': '069500', 'KODEX 코스닥150': '229200',
    'TIGER 200': '102110', 'KODEX 레버리지': '122630',
    'KODEX 인버스': '114800', 'KODEX 반도체': '091160',
    'TIGER 미국S&P500': '360750', 'KODEX 미국나스닥100': '379810',
    'KODEX 배당가치': '290080', 'KODEX 고배당': '279530',
}

DEFAULT_TICKERS = """삼성전자,005930
SK하이닉스,000660
NAVER,035420
카카오,035720
현대차,005380
LG에너지솔루션,373220
KB금융,105560
KODEX 200,069500
KODEX 반도체,091160
TIGER 미국S&P500,360750"""


# ══════════════════════════════════════════
# Shared Data
# ══════════════════════════════════════════
class DataStore:
    price_df = None
    returns = None
    tickers = {}
    risk_df = None
    opt_results = {}
    cov_annual = None
    mu_annual = None


data = DataStore()


# ══════════════════════════════════════════
# Helper / Computation Functions
# ══════════════════════════════════════════
def get_cache_dir():
    try:
        app = MDApp.get_running_app()
        d = os.path.join(app.user_data_dir, 'charts')
    except Exception:
        d = os.path.join(tempfile.gettempdir(), 'portfolio_charts')
    os.makedirs(d, exist_ok=True)
    return d


def save_chart(fig, name):
    path = os.path.join(get_cache_dir(), f'{name}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def fetch_stock_data(tickers_dict, years=5):
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    all_data = {}
    errors = []
    for name, code in tickers_dict.items():
        try:
            code = str(code).strip().zfill(6)
            for suffix in ['.KS', '.KQ']:
                ticker = f"{code}{suffix}"
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df is not None and len(df) > 60:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    all_data[name] = df['Close']
                    break
            else:
                errors.append(name)
        except Exception:
            errors.append(name)
    if all_data:
        price_df = pd.DataFrame(all_data).dropna(how='all').ffill()
        return price_df, errors
    return None, errors


def calc_var(r, conf=0.95):
    return np.percentile(r.dropna(), (1 - conf) * 100)


def calc_cvar(r, conf=0.95):
    v = calc_var(r, conf)
    return r[r <= v].mean()


def calc_mdd(prices):
    cm = prices.cummax()
    return ((prices - cm) / cm).min()


def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)


def portfolio_ret(w, mu):
    return w @ mu


def _project_weights(w, max_w=0.3, min_w=0.0):
    """Project weights onto constraints: sum=1, min_w <= w_i <= max_w."""
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.ones(len(w)) / len(w)
    # Re-clip and normalize iteratively
    for _ in range(50):
        w = np.clip(w, min_w, max_w)
        s = w.sum()
        if abs(s - 1.0) < 1e-10:
            break
        w = w / s
    return w


def optimize_portfolio(mu, cov, n, method='mvp', max_w=0.3, min_w=0.0, rf=0.035):
    """Portfolio optimization using projected gradient descent (no scipy)."""
    w = np.ones(n) / n
    lr = 0.01
    try:
        for iteration in range(2000):
            if method == 'mvp':
                grad = cov @ w / portfolio_vol(w, cov)
            elif method == 'max_sharpe':
                pv = portfolio_vol(w, cov)
                pr = portfolio_ret(w, mu) - rf
                grad_vol = cov @ w / pv
                grad_ret = mu
                sr = pr / pv
                grad = -(grad_ret * pv - pr * grad_vol) / (pv ** 2)
            elif method == 'risk_parity':
                pv = portfolio_vol(w, cov)
                mr = cov @ w
                rc = w * mr / pv
                target = pv / n
                diff = rc - target
                # Approximate gradient
                grad = np.zeros(n)
                eps = 1e-6
                for i in range(n):
                    w_p = w.copy()
                    w_p[i] += eps
                    w_p = w_p / w_p.sum()
                    pv_p = portfolio_vol(w_p, cov)
                    mr_p = cov @ w_p
                    rc_p = w_p * mr_p / pv_p
                    obj_p = np.sum((rc_p - pv_p / n) ** 2)
                    obj_c = np.sum(diff ** 2)
                    grad[i] = (obj_p - obj_c) / eps
            else:
                return w

            w = w - lr * grad
            w = _project_weights(w, max_w, min_w)

            # Adaptive learning rate
            if iteration % 500 == 499:
                lr *= 0.5

        return w
    except Exception:
        return np.ones(n) / n


# ══════════════════════════════════════════
# KV Layout
# ══════════════════════════════════════════
KV = '''
#:import dp kivy.metrics.dp

<DrawerItem@OneLineIconListItem>:
    icon: "home"
    on_release: app.switch_screen(self.screen_name)
    screen_name: ""
    IconLeftWidget:
        icon: root.icon

MDNavigationLayout:
    MDScreenManager:
        id: screen_manager

        HomeScreen:
            name: "home"
        StockSelectScreen:
            name: "stock"
        RiskScreen:
            name: "risk"
        OptScreen:
            name: "opt"
        FrontierScreen:
            name: "frontier"
        BacktestScreen:
            name: "backtest"
        RebalScreen:
            name: "rebal"

    MDNavigationDrawer:
        id: nav_drawer
        radius: (0, dp(16), dp(16), 0)

        BoxLayout:
            orientation: "vertical"
            padding: dp(8)
            spacing: dp(4)

            MDLabel:
                text: "포트폴리오 최적화"
                font_style: "H6"
                size_hint_y: None
                height: dp(56)
                padding: [dp(16), dp(8)]

            MDLabel:
                text: "리스크 최소화 전략"
                theme_text_color: "Secondary"
                size_hint_y: None
                height: dp(24)
                padding: [dp(16), 0]

            Widget:
                size_hint_y: None
                height: dp(8)

            ScrollView:
                MDList:
                    DrawerItem:
                        icon: "home"
                        text: "홈"
                        screen_name: "home"
                    DrawerItem:
                        icon: "database"
                        text: "① 종목선택·데이터"
                        screen_name: "stock"
                    DrawerItem:
                        icon: "alert-circle-outline"
                        text: "② 리스크분석"
                        screen_name: "risk"
                    DrawerItem:
                        icon: "target"
                        text: "③ 포트폴리오최적화"
                        screen_name: "opt"
                    DrawerItem:
                        icon: "chart-bell-curve-cumulative"
                        text: "④ 효율적프론티어"
                        screen_name: "frontier"
                    DrawerItem:
                        icon: "history"
                        text: "⑤ 백테스트"
                        screen_name: "backtest"
                    DrawerItem:
                        icon: "refresh"
                        text: "⑥ 리밸런싱"
                        screen_name: "rebal"


# ─── Home ───
<HomeScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "포트폴리오 최적화"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(20)
                spacing: dp(16)

                Widget:
                    size_hint_y: None
                    height: dp(16)

                MDLabel:
                    text: "리스크 최소화\\n주식 포트폴리오 최적화 시스템"
                    font_style: "H5"
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                MDLabel:
                    text: "KOSPI/KOSDAQ 종목과 ETF 대상\\n투자 리스크를 최소화하는 최적 포트폴리오 산출"
                    halign: "center"
                    theme_text_color: "Secondary"
                    size_hint_y: None
                    height: self.texture_size[1]

                Widget:
                    size_hint_y: None
                    height: dp(8)

                MDCard:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16)
                    spacing: dp(6)
                    radius: [dp(12)]
                    elevation: 1
                    md_bg_color: app.theme_cls.bg_darkest

                    MDLabel:
                        text: "주요 기능"
                        font_style: "H6"
                        size_hint_y: None
                        height: self.texture_size[1]

                    MDLabel:
                        text: "① 종목 선택 · 데이터 수집\\n② 개별 종목 리스크 분석\\n③ 4가지 포트폴리오 최적화\\n④ 효율적 프론티어 시각화\\n⑤ 백테스트 (과거 성과 검증)\\n⑥ 리밸런싱 시뮬레이션"
                        size_hint_y: None
                        height: self.texture_size[1]

                MDCard:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    padding: dp(16)
                    spacing: dp(6)
                    radius: [dp(12)]
                    elevation: 1
                    md_bg_color: app.theme_cls.bg_darkest

                    MDLabel:
                        text: "설정"
                        font_style: "H6"
                        size_hint_y: None
                        height: self.texture_size[1]

                    MDLabel:
                        text: "투자 대상: 개별 종목 + ETF\\n종목 수: 10~30\\n데이터 기간: 최대 5년\\n개별 비중 상한: 30%\\n리밸런싱: 월간/분기"
                        size_hint_y: None
                        height: self.texture_size[1]

                MDRaisedButton:
                    text: "시작하기 → 종목 선택"
                    pos_hint: {"center_x": .5}
                    on_release: app.switch_screen("stock")

                MDLabel:
                    text: "금오공과대학교 경영학과 | 2026"
                    halign: "center"
                    theme_text_color: "Hint"
                    size_hint_y: None
                    height: self.texture_size[1]

                Widget:
                    size_hint_y: None
                    height: dp(40)


# ─── Stock Select ───
<StockSelectScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "① 종목 선택·데이터"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: stock_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDLabel:
                    text: "종목 입력 (종목명,코드)"
                    font_style: "Subtitle1"
                    size_hint_y: None
                    height: self.texture_size[1]

                MDTextField:
                    id: ticker_input
                    hint_text: "종목명,코드 (줄바꿈 구분)"
                    mode: "rectangle"
                    multiline: True
                    size_hint_y: None
                    height: dp(200)

                MDLabel:
                    text: "인기 종목 빠른 선택"
                    font_style: "Subtitle1"
                    size_hint_y: None
                    height: self.texture_size[1]

                GridLayout:
                    id: preset_grid
                    cols: 2
                    spacing: dp(6)
                    size_hint_y: None
                    height: self.minimum_height

                MDRaisedButton:
                    text: "데이터 수집 시작"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.fetch_data()

                MDLabel:
                    id: stock_status
                    text: ""
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: stock_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height


# ─── Risk ───
<RiskScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "② 리스크 분석"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: risk_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDRaisedButton:
                    text: "리스크 분석 실행"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.run_analysis()

                MDLabel:
                    id: risk_status
                    text: "데이터를 먼저 수집해 주세요"
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: risk_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height


# ─── Optimization ───
<OptScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "③ 포트폴리오 최적화"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: opt_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDLabel:
                    text: "비중 상한: 30%  |  무위험이자율: 3.5%"
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                MDRaisedButton:
                    text: "4가지 최적화 실행"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.run_optimization()

                MDLabel:
                    id: opt_status
                    text: ""
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: opt_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height


# ─── Frontier ───
<FrontierScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "④ 효율적 프론티어"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: frontier_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDRaisedButton:
                    text: "효율적 프론티어 산출"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.run_frontier()

                MDLabel:
                    id: frontier_status
                    text: ""
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: frontier_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height


# ─── Backtest ───
<BacktestScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "⑤ 백테스트"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: bt_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDLabel:
                    text: "학습: 1년 | MVP 전략 | 월간 리밸런싱"
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                MDRaisedButton:
                    text: "백테스트 실행"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.run_backtest()

                MDLabel:
                    id: bt_status
                    text: ""
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: bt_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height


# ─── Rebalancing ───
<RebalScreen>:
    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "⑥ 리밸런싱"
            left_action_items: [["menu", lambda x: app.toggle_nav()]]
            elevation: 2

        ScrollView:
            MDBoxLayout:
                id: rebal_content
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                spacing: dp(12)

                MDTextField:
                    id: invest_input
                    hint_text: "총 투자금 (만원)"
                    text: "10000"
                    mode: "rectangle"
                    input_filter: "int"
                    size_hint_y: None
                    height: dp(48)

                MDRaisedButton:
                    text: "리밸런싱 분석 실행"
                    pos_hint: {"center_x": .5}
                    size_hint_x: 1
                    on_release: root.run_rebalancing()

                MDLabel:
                    id: rebal_status
                    text: ""
                    halign: "center"
                    size_hint_y: None
                    height: self.texture_size[1]

                BoxLayout:
                    id: rebal_results
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
'''


# ══════════════════════════════════════════
# Chart helper
# ══════════════════════════════════════════
def add_chart_to(parent, path):
    """Add a chart image to a parent layout."""
    img = KivyImage(
        source=path,
        size_hint_y=None,
        height=dp(300),
        allow_stretch=True,
        keep_ratio=True,
    )
    parent.add_widget(img)


def add_label_to(parent, text, **kwargs):
    lbl = MDLabel(
        text=text,
        size_hint_y=None,
        **kwargs,
    )
    lbl.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))
    parent.add_widget(lbl)


def add_card_text(parent, title, body):
    card = MDCard(
        orientation="vertical",
        size_hint_y=None,
        padding=dp(12),
        spacing=dp(4),
        radius=[dp(10)],
        elevation=1,
    )
    card.bind(minimum_height=card.setter('height'))
    t = MDLabel(text=title, font_style="Subtitle1", bold=True,
                size_hint_y=None)
    t.bind(texture_size=lambda i, v: setattr(i, 'height', v[1]))
    b = MDLabel(text=body, size_hint_y=None, theme_text_color="Secondary")
    b.bind(texture_size=lambda i, v: setattr(i, 'height', v[1]))
    card.add_widget(t)
    card.add_widget(b)
    parent.add_widget(card)


def clear_box(box):
    box.clear_widgets()


# ══════════════════════════════════════════
# Screen Classes
# ══════════════════════════════════════════

class HomeScreen(Screen):
    pass


class StockSelectScreen(Screen):

    def on_enter(self):
        inp = self.ids.ticker_input
        if not inp.text:
            inp.text = DEFAULT_TICKERS
        # preset buttons
        grid = self.ids.preset_grid
        if len(grid.children) == 0:
            presets = [
                ("대형주 10종", "삼성전자,005930\nSK하이닉스,000660\nNAVER,035420\n현대차,005380\n카카오,035720\nLG에너지솔루션,373220\nKB금융,105560\n삼성SDI,006400\nLG화학,051910\nPOSCO홀딩스,005490"),
                ("ETF 10종", "KODEX 200,069500\nKODEX 반도체,091160\nTIGER 미국S&P500,360750\nKODEX 미국나스닥100,379810\nKODEX 레버리지,122630\nKODEX 인버스,114800\nKODEX 배당가치,290080\nKODEX 고배당,279530\nTIGER 200,102110\nKODEX 코스닥150,229200"),
                ("혼합 10종 (기본)", DEFAULT_TICKERS),
            ]
            for label, val in presets:
                btn = MDFlatButton(
                    text=label,
                    size_hint_y=None,
                    height=dp(40),
                    on_release=lambda x, v=val: self._set_input(v),
                )
                grid.add_widget(btn)

    def _set_input(self, val):
        self.ids.ticker_input.text = val

    def fetch_data(self):
        if not HAS_YFINANCE:
            self.ids.stock_status.text = "yfinance 패키지가 없습니다."
            return
        text = self.ids.ticker_input.text.strip()
        if not text:
            self.ids.stock_status.text = "종목을 입력해 주세요."
            return

        tickers = {}
        for line in text.split('\n'):
            line = line.strip()
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    name, code = parts[0].strip(), parts[1].strip()
                    if name and code:
                        tickers[name] = code
        if len(tickers) < 2:
            self.ids.stock_status.text = "최소 2개 이상 종목을 입력하세요."
            return

        self.ids.stock_status.text = f"{len(tickers)}개 종목 데이터 수집 중..."
        threading.Thread(target=self._fetch_thread, args=(tickers,), daemon=True).start()

    def _fetch_thread(self, tickers):
        try:
            price_df, errors = fetch_stock_data(tickers, years=5)
            Clock.schedule_once(lambda dt: self._on_fetch_done(price_df, errors, tickers))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._on_fetch_error(str(e)))

    def _on_fetch_error(self, msg):
        self.ids.stock_status.text = f"오류: {msg}"

    def _on_fetch_done(self, price_df, errors, tickers):
        results = self.ids.stock_results
        clear_box(results)

        if price_df is None or len(price_df.columns) < 2:
            self.ids.stock_status.text = "데이터 수집 실패. 종목코드를 확인하세요."
            return

        data.price_df = price_df
        data.returns = price_df.pct_change().dropna()
        data.tickers = {k: v for k, v in tickers.items() if k in price_df.columns}

        status = f"{len(price_df.columns)}개 종목, {len(price_df)}거래일 수집 완료!"
        if errors:
            status += f"\n실패: {', '.join(errors)}"
        self.ids.stock_status.text = status

        # Summary card
        start_d = price_df.index[0].strftime('%Y-%m-%d')
        end_d = price_df.index[-1].strftime('%Y-%m-%d')
        add_card_text(results, "데이터 요약",
                      f"종목: {', '.join(price_df.columns)}\n"
                      f"기간: {start_d} ~ {end_d}\n"
                      f"거래일: {len(price_df):,}일")

        # Return stats
        returns = data.returns
        stats_lines = []
        for col in returns.columns:
            ann_r = returns[col].mean() * 252 * 100
            ann_v = returns[col].std() * np.sqrt(252) * 100
            sr = (ann_r / 100) / (ann_v / 100) if ann_v > 0 else 0
            stats_lines.append(f"{col}: 수익률 {ann_r:.1f}% | 변동성 {ann_v:.1f}% | 샤프 {sr:.2f}")
        add_card_text(results, "수익률 통계", '\n'.join(stats_lines))

        # Normalized price chart
        try:
            norm = price_df / price_df.iloc[0] * 100
            fig, ax = plt.subplots(figsize=(8, 4))
            for col in norm.columns:
                ax.plot(norm.index, norm[col], label=col, linewidth=1)
            ax.set_title('Normalized Price (Start=100)')
            ax.set_ylabel('Price')
            ax.legend(fontsize=6, ncol=3, loc='upper left')
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            path = save_chart(fig, 'price_norm')
            add_chart_to(results, path)
        except Exception:
            pass


class RiskScreen(Screen):

    def run_analysis(self):
        if data.returns is None:
            self.ids.risk_status.text = "먼저 종목선택에서 데이터를 수집하세요."
            return
        self.ids.risk_status.text = "분석 중..."
        threading.Thread(target=self._analysis_thread, daemon=True).start()

    def _analysis_thread(self):
        try:
            returns = data.returns
            price_df = data.price_df
            conf = 0.95
            risk_rows = []

            for col in returns.columns:
                r = returns[col].dropna()
                p = price_df[col].dropna()
                ann_v = r.std() * np.sqrt(252)
                ann_r = r.mean() * 252
                sr = ann_r / ann_v if ann_v > 0 else 0
                var_val = calc_var(r, conf)
                cvar_val = calc_cvar(r, conf)
                mdd = calc_mdd(p)
                risk_rows.append({
                    'name': col,
                    'ann_ret': ann_r * 100,
                    'ann_vol': ann_v * 100,
                    'sharpe': sr,
                    'var': var_val * 100,
                    'cvar': cvar_val * 100,
                    'mdd': mdd * 100,
                })

            data.risk_df = pd.DataFrame(risk_rows)
            corr = returns.corr()

            # Chart 1: Volatility bar
            fig1, ax1 = plt.subplots(figsize=(8, max(3, len(risk_rows) * 0.35)))
            sorted_r = sorted(risk_rows, key=lambda x: x['ann_vol'])
            names = [r['name'] for r in sorted_r]
            vols = [r['ann_vol'] for r in sorted_r]
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(vols)))
            ax1.barh(names, vols, color=colors)
            for i, v in enumerate(vols):
                ax1.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=7)
            ax1.set_title('Annual Volatility (%)')
            ax1.set_xlabel('Volatility (%)')
            fig1.tight_layout()
            path1 = save_chart(fig1, 'risk_vol')

            # Chart 2: Risk-Return scatter
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            xs = [r['ann_vol'] for r in risk_rows]
            ys = [r['ann_ret'] for r in risk_rows]
            cs = [r['sharpe'] for r in risk_rows]
            sc = ax2.scatter(xs, ys, c=cs, cmap='RdYlGn', s=80, edgecolors='gray')
            for r in risk_rows:
                ax2.annotate(r['name'], (r['ann_vol'], r['ann_ret']),
                             fontsize=6, ha='center', va='bottom')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('Return vs Volatility')
            ax2.set_xlabel('Annual Volatility (%)')
            ax2.set_ylabel('Annual Return (%)')
            plt.colorbar(sc, label='Sharpe Ratio')
            fig2.tight_layout()
            path2 = save_chart(fig2, 'risk_scatter')

            # Chart 3: Correlation heatmap
            fig3, ax3 = plt.subplots(figsize=(8, max(5, len(corr) * 0.4)))
            im = ax3.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax3.set_xticks(range(len(corr)))
            ax3.set_yticks(range(len(corr)))
            ax3.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=6)
            ax3.set_yticklabels(corr.index, fontsize=6)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax3.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=5)
            plt.colorbar(im)
            ax3.set_title('Correlation Matrix')
            fig3.tight_layout()
            path3 = save_chart(fig3, 'risk_corr')

            Clock.schedule_once(lambda dt: self._show_results(risk_rows, path1, path2, path3))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.ids.risk_status, 'text', f'오류: {e}'))

    def _show_results(self, risk_rows, path1, path2, path3):
        box = self.ids.risk_results
        clear_box(box)
        self.ids.risk_status.text = "분석 완료!"

        # Risk table
        lines = []
        for r in risk_rows:
            lines.append(
                f"{r['name']}\n"
                f"  수익률 {r['ann_ret']:.1f}% | 변동성 {r['ann_vol']:.1f}% | "
                f"샤프 {r['sharpe']:.2f}\n"
                f"  VaR {r['var']:.2f}% | CVaR {r['cvar']:.2f}% | "
                f"MDD {r['mdd']:.1f}%"
            )
        add_card_text(box, "종목별 리스크 지표", '\n\n'.join(lines))

        add_label_to(box, "연간 변동성 비교", font_style="Subtitle1", bold=True)
        add_chart_to(box, path1)

        add_label_to(box, "수익률 vs 변동성", font_style="Subtitle1", bold=True)
        add_chart_to(box, path2)

        add_label_to(box, "상관행렬", font_style="Subtitle1", bold=True)
        add_chart_to(box, path3)


class OptScreen(Screen):

    def run_optimization(self):
        if data.returns is None:
            self.ids.opt_status.text = "먼저 데이터를 수집하세요."
            return
        self.ids.opt_status.text = "최적화 중..."
        threading.Thread(target=self._opt_thread, daemon=True).start()

    def _opt_thread(self):
        try:
            returns = data.returns
            n = len(returns.columns)
            names = list(returns.columns)
            mu = returns.mean().values * 252
            cov = returns.cov().values * 252
            rf = 0.035
            max_w = 0.3

            data.mu_annual = mu
            data.cov_annual = cov

            results = {}
            for method, label in [('mvp', '최소분산(MVP)'), ('max_sharpe', '최대샤프비율'),
                                  ('risk_parity', '리스크패리티')]:
                w = optimize_portfolio(mu, cov, n, method=method, max_w=max_w, rf=rf)
                results[label] = w
            results['균등배분'] = np.ones(n) / n

            data.opt_results = results

            # Compare
            compare = []
            for label, w in results.items():
                ret = portfolio_ret(w, mu)
                vol = portfolio_vol(w, cov)
                sr = (ret - rf) / vol if vol > 0 else 0
                compare.append({
                    'model': label, 'ret': ret * 100, 'vol': vol * 100,
                    'sharpe': sr, 'max_w': w.max() * 100,
                    'n_stocks': int(np.sum(w > 0.01)),
                })

            # Chart: model positions
            fig, ax = plt.subplots(figsize=(8, 5))
            colors_m = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
            for i, c in enumerate(compare):
                ax.scatter(c['vol'], c['ret'], s=200, c=colors_m[i % 4],
                           label=c['model'], zorder=5, edgecolors='black')
                ax.annotate(c['model'], (c['vol'], c['ret']),
                            fontsize=7, ha='center', va='bottom')
            ax.set_title('Model Risk-Return')
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Return (%)')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path_cmp = save_chart(fig, 'opt_compare')

            # Pie charts for each model
            pie_paths = {}
            for label, w in results.items():
                fig_p, ax_p = plt.subplots(figsize=(6, 6))
                mask = w > 0.01
                ax_p.pie(w[mask], labels=[names[i] for i in range(n) if mask[i]],
                         autopct='%1.1f%%', textprops={'fontsize': 7})
                ax_p.set_title(f'{label}')
                fig_p.tight_layout()
                safe_label = label.replace('(', '').replace(')', '')
                pie_paths[label] = save_chart(fig_p, f'opt_pie_{safe_label}')

            Clock.schedule_once(lambda dt: self._show_results(compare, names, results, path_cmp, pie_paths))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.ids.opt_status, 'text', f'오류: {e}'))

    def _show_results(self, compare, names, results, path_cmp, pie_paths):
        box = self.ids.opt_results
        clear_box(box)
        self.ids.opt_status.text = f"{len(results)}개 모형 최적화 완료!"

        # Compare table
        lines = []
        for c in compare:
            lines.append(
                f"{c['model']}\n"
                f"  수익률 {c['ret']:.2f}% | 변동성 {c['vol']:.2f}% | "
                f"샤프 {c['sharpe']:.3f}\n"
                f"  최대비중 {c['max_w']:.1f}% | 편입 {c['n_stocks']}종목"
            )
        add_card_text(box, "모형별 비교", '\n\n'.join(lines))

        add_label_to(box, "모형별 위험-수익 위치", font_style="Subtitle1", bold=True)
        add_chart_to(box, path_cmp)

        # Weight details
        for label, w in results.items():
            n = len(w)
            weight_lines = []
            indices = np.argsort(-w)
            for idx in indices:
                if w[idx] > 0.01:
                    weight_lines.append(f"  {names[idx]}: {w[idx]*100:.1f}%")
            add_card_text(box, f"{label} 비중", '\n'.join(weight_lines))
            if label in pie_paths:
                add_chart_to(box, pie_paths[label])


class FrontierScreen(Screen):

    def run_frontier(self):
        if data.returns is None:
            self.ids.frontier_status.text = "먼저 데이터를 수집하세요."
            return
        self.ids.frontier_status.text = "프론티어 계산 중..."
        threading.Thread(target=self._frontier_thread, daemon=True).start()

    def _frontier_thread(self):
        try:
            returns = data.returns
            n = len(returns.columns)
            names = list(returns.columns)
            mu = returns.mean().values * 252
            cov = returns.cov().values * 252
            rf = 0.035
            max_w = 0.3
            n_points = 60

            w0 = np.ones(n) / n
            bounds = [(0, max_w)] * n
            cons_eq = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            # Min variance
            res_min = minimize(lambda w: portfolio_vol(w, cov), w0,
                               method='SLSQP', bounds=bounds, constraints=cons_eq)
            min_v = portfolio_vol(res_min.x, cov)
            min_r = portfolio_ret(res_min.x, mu)

            # Max return
            res_max = minimize(lambda w: -portfolio_ret(w, mu), w0,
                               method='SLSQP', bounds=bounds, constraints=cons_eq)
            max_r = portfolio_ret(res_max.x, mu)

            # Frontier points
            targets = np.linspace(min_r, max_r, n_points)
            f_vols, f_rets = [], []
            for t in targets:
                c = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                     {'type': 'eq', 'fun': lambda w, tt=t: portfolio_ret(w, mu) - tt}]
                try:
                    res = minimize(lambda w: portfolio_vol(w, cov), w0,
                                   method='SLSQP', bounds=bounds, constraints=c)
                    if res.success:
                        f_vols.append(portfolio_vol(res.x, cov) * 100)
                        f_rets.append(portfolio_ret(res.x, mu) * 100)
                except Exception:
                    pass

            # Max Sharpe
            def neg_sr(w):
                return -(portfolio_ret(w, mu) - rf) / portfolio_vol(w, cov)
            res_sr = minimize(neg_sr, w0, method='SLSQP', bounds=bounds, constraints=cons_eq)
            sr_v = portfolio_vol(res_sr.x, cov) * 100
            sr_r = portfolio_ret(res_sr.x, mu) * 100

            # Random portfolios
            rand_v, rand_r, rand_s = [], [], []
            for _ in range(2000):
                rw = np.random.random(n)
                rw = np.minimum(rw, max_w)
                rw /= rw.sum()
                rv = portfolio_vol(rw, cov) * 100
                rr = portfolio_ret(rw, mu) * 100
                rand_v.append(rv)
                rand_r.append(rr)
                rand_s.append((rr / 100 - rf) / (rv / 100) if rv > 0 else 0)

            # Chart
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(rand_v, rand_r, c=rand_s, cmap='viridis', s=3, alpha=0.4)
            plt.colorbar(sc, label='Sharpe Ratio', ax=ax)
            ax.plot(f_vols, f_rets, 'r-', linewidth=2.5, label='Efficient Frontier')
            ax.scatter([min_v * 100], [min_r * 100], marker='*', s=250, c='blue',
                       label='MVP', zorder=5)
            ax.scatter([sr_v], [sr_r], marker='D', s=150, c='green',
                       label='Max Sharpe', zorder=5)

            # Individual stocks
            for i in range(n):
                vi = np.sqrt(cov[i, i]) * 100
                ri = mu[i] * 100
                ax.scatter([vi], [ri], s=30, c='black', zorder=4)
                ax.annotate(names[i], (vi, ri), fontsize=5, ha='center', va='bottom')

            # CML
            sharpe_ratio = (sr_r / 100 - rf) / (sr_v / 100) if sr_v > 0 else 0
            cml_x = np.linspace(0, max(f_vols) * 1.1, 50)
            cml_y = (rf + sharpe_ratio * cml_x / 100) * 100
            ax.plot(cml_x, cml_y, '--', color='orange', linewidth=1.5, label='CML')

            ax.set_title('Efficient Frontier & CML')
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Return (%)')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = save_chart(fig, 'frontier')

            summary = (f"MVP: Vol {min_v*100:.2f}%, Ret {min_r*100:.2f}%\n"
                       f"Max Sharpe: Vol {sr_v:.2f}%, Ret {sr_r:.2f}%\n"
                       f"Sharpe Ratio: {sharpe_ratio:.3f}")

            Clock.schedule_once(lambda dt: self._show_results(path, summary))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.ids.frontier_status, 'text', f'오류: {e}'))

    def _show_results(self, path, summary):
        box = self.ids.frontier_results
        clear_box(box)
        self.ids.frontier_status.text = "프론티어 산출 완료!"

        add_card_text(box, "핵심 포트폴리오", summary)
        add_label_to(box, "효율적 프론티어", font_style="Subtitle1", bold=True)
        add_chart_to(box, path)


class BacktestScreen(Screen):

    def run_backtest(self):
        if data.returns is None:
            self.ids.bt_status.text = "먼저 데이터를 수집하세요."
            return
        self.ids.bt_status.text = "백테스트 실행 중..."
        threading.Thread(target=self._bt_thread, daemon=True).start()

    def _bt_thread(self):
        try:
            returns = data.returns
            price_df = data.price_df
            n = len(returns.columns)
            lb_days = 252
            rebal_days = 21
            max_w = 0.3

            start_idx = lb_days
            if start_idx >= len(returns):
                Clock.schedule_once(lambda dt: setattr(self.ids.bt_status, 'text', '데이터 부족'))
                return

            dates = returns.index[start_idx:]
            port_vals = [1.0]
            equal_vals = [1.0]
            current_w = np.ones(n) / n
            last_rebal = 0

            for i, date in enumerate(dates):
                idx = start_idx + i
                if i - last_rebal >= rebal_days or i == 0:
                    train = returns.iloc[max(0, idx - lb_days):idx]
                    cov_t = train.cov().values * 252
                    mu_t = train.mean().values * 252
                    current_w = optimize_portfolio(mu_t, cov_t, n, 'mvp', max_w)
                    last_rebal = i

                daily = returns.iloc[idx].values
                port_vals.append(port_vals[-1] * (1 + current_w @ daily))
                equal_vals.append(equal_vals[-1] * (1 + np.ones(n) / n @ daily))

            port_s = pd.Series(port_vals[1:], index=dates)
            equal_s = pd.Series(equal_vals[1:], index=dates)

            # Metrics
            def calc_bt_metrics(s):
                d = s.pct_change().dropna()
                yrs = len(d) / 252
                ann_r = ((s.iloc[-1]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
                ann_v = d.std() * np.sqrt(252) * 100
                sr = (ann_r / 100 - 0.035) / (ann_v / 100) if ann_v > 0 else 0
                mdd = ((s - s.cummax()) / s.cummax()).min() * 100
                return {'ann_r': ann_r, 'ann_v': ann_v, 'sharpe': sr, 'mdd': mdd,
                        'total': (s.iloc[-1] - 1) * 100}

            pm = calc_bt_metrics(port_s)
            em = calc_bt_metrics(equal_s)

            # Chart 1: cumulative
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(dates, (port_s - 1) * 100, color='#E53935', linewidth=2, label='MVP')
            ax1.plot(dates, (equal_s - 1) * 100, color='#1E88E5', linewidth=1.5,
                     linestyle='--', label='Equal Weight')
            ax1.set_title('Cumulative Return (%)')
            ax1.set_ylabel('Return (%)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            fig1.autofmt_xdate()
            fig1.tight_layout()
            path1 = save_chart(fig1, 'bt_cumulative')

            # Chart 2: drawdown
            dd = (port_s - port_s.cummax()) / port_s.cummax() * 100
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.fill_between(dates, dd, 0, color='#E53935', alpha=0.4)
            ax2.plot(dates, dd, color='#E53935', linewidth=1)
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            fig2.autofmt_xdate()
            fig2.tight_layout()
            path2 = save_chart(fig2, 'bt_drawdown')

            Clock.schedule_once(lambda dt: self._show_results(pm, em, path1, path2))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.ids.bt_status, 'text', f'오류: {e}'))

    def _show_results(self, pm, em, path1, path2):
        box = self.ids.bt_results
        clear_box(box)
        self.ids.bt_status.text = "백테스트 완료!"

        metrics_text = (
            f"최소분산(MVP)\n"
            f"  총수익률: {pm['total']:.2f}%\n"
            f"  연환산수익률: {pm['ann_r']:.2f}%\n"
            f"  연간변동성: {pm['ann_v']:.2f}%\n"
            f"  샤프비율: {pm['sharpe']:.3f}\n"
            f"  MDD: {pm['mdd']:.2f}%\n\n"
            f"균등배분\n"
            f"  총수익률: {em['total']:.2f}%\n"
            f"  연환산수익률: {em['ann_r']:.2f}%\n"
            f"  연간변동성: {em['ann_v']:.2f}%\n"
            f"  샤프비율: {em['sharpe']:.3f}\n"
            f"  MDD: {em['mdd']:.2f}%"
        )
        add_card_text(box, "성과 지표", metrics_text)

        add_label_to(box, "누적 수익률", font_style="Subtitle1", bold=True)
        add_chart_to(box, path1)

        add_label_to(box, "낙폭 (Drawdown)", font_style="Subtitle1", bold=True)
        add_chart_to(box, path2)


class RebalScreen(Screen):

    def run_rebalancing(self):
        if data.returns is None:
            self.ids.rebal_status.text = "먼저 데이터를 수집하세요."
            return
        try:
            total = int(self.ids.invest_input.text)
        except ValueError:
            total = 10000
        self.ids.rebal_status.text = "분석 중..."
        threading.Thread(target=self._rebal_thread, args=(total,), daemon=True).start()

    def _rebal_thread(self, total_invest):
        try:
            returns = data.returns
            price_df = data.price_df
            n = len(returns.columns)
            names = list(returns.columns)
            mu = returns.mean().values * 252
            cov = returns.cov().values * 252
            max_w = 0.3

            opt_w = optimize_portfolio(mu, cov, n, 'mvp', max_w)

            # Weight table with buy amounts
            last_prices = price_df.iloc[-1]
            lines = []
            for idx in np.argsort(-opt_w):
                w = opt_w[idx]
                if w < 0.005:
                    continue
                name = names[idx]
                invest = w * total_invest
                price = last_prices[name] if name in last_prices.index else 0
                qty = int(invest * 10000 / price) if price > 0 else 0
                actual = qty * price / 10000
                lines.append(
                    f"{name}\n"
                    f"  비중 {w*100:.1f}% | 투자 {invest:.0f}만원 | "
                    f"현재가 {price:,.0f}원\n"
                    f"  매수 {qty}주 | 실투자 {actual:.1f}만원"
                )
            weight_text = '\n\n'.join(lines)

            # Rebal comparison
            lb_days = 252
            rebal_days = 21
            start_idx = lb_days

            rebal_vals = [1.0]
            hold_vals = [1.0]
            w_rebal = opt_w.copy()
            w_hold = opt_w.copy()
            last_r = 0

            if start_idx < len(returns):
                dates = returns.index[start_idx:]
                for i, date in enumerate(dates):
                    idx = start_idx + i
                    daily = returns.iloc[idx].values

                    if i - last_r >= rebal_days or i == 0:
                        w_rebal = optimize_portfolio(mu, cov, n, 'mvp', max_w)
                        last_r = i

                    rebal_vals.append(rebal_vals[-1] * (1 + w_rebal @ daily))
                    hold_vals.append(hold_vals[-1] * (1 + w_hold @ daily))
                    w_hold = w_hold * (1 + daily)
                    w_hold = w_hold / w_hold.sum()

                rebal_s = pd.Series(rebal_vals[1:], index=dates)
                hold_s = pd.Series(hold_vals[1:], index=dates)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dates, (rebal_s - 1) * 100, color='#E53935', linewidth=2,
                        label='Rebalancing O')
                ax.plot(dates, (hold_s - 1) * 100, color='#1E88E5', linewidth=1.5,
                        linestyle='--', label='Buy & Hold')
                ax.set_title('Rebalancing Effect')
                ax.set_ylabel('Return (%)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                fig.tight_layout()
                path_cmp = save_chart(fig, 'rebal_compare')
            else:
                path_cmp = None

            # Pie chart
            fig_p, ax_p = plt.subplots(figsize=(6, 6))
            mask = opt_w > 0.01
            ax_p.pie(opt_w[mask],
                     labels=[names[i] for i in range(n) if mask[i]],
                     autopct='%1.1f%%', textprops={'fontsize': 7})
            ax_p.set_title('MVP Optimal Weights')
            fig_p.tight_layout()
            path_pie = save_chart(fig_p, 'rebal_pie')

            actual_total = sum(
                int(opt_w[i] * total_invest * 10000 / last_prices[names[i]]) * last_prices[names[i]] / 10000
                for i in range(n) if opt_w[i] > 0.005 and names[i] in last_prices.index and last_prices[names[i]] > 0
            )
            remaining = total_invest - actual_total

            summary = (f"총 투자금: {total_invest:,}만원\n"
                       f"실 투자금: {actual_total:,.1f}만원\n"
                       f"잔여 현금: {remaining:,.1f}만원\n"
                       f"편입 종목: {int(np.sum(opt_w > 0.005))}개")

            Clock.schedule_once(lambda dt: self._show_results(
                summary, weight_text, path_pie, path_cmp))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.ids.rebal_status, 'text', f'오류: {e}'))

    def _show_results(self, summary, weight_text, path_pie, path_cmp):
        box = self.ids.rebal_results
        clear_box(box)
        self.ids.rebal_status.text = "분석 완료!"

        add_card_text(box, "투자 요약", summary)
        add_card_text(box, "최소분산(MVP) 매수 안내", weight_text)
        add_chart_to(box, path_pie)

        if path_cmp:
            add_label_to(box, "리밸런싱 효과 비교", font_style="Subtitle1", bold=True)
            add_chart_to(box, path_cmp)

        add_card_text(box, "리밸런싱 체크리스트",
                      "1. 현재 보유 종목 평가금액 확인\n"
                      "2. 이 앱에서 최적 비중 재산출\n"
                      "3. 목표 비중과 현재 비중 차이 계산\n"
                      "4. 비중 초과 종목 매도\n"
                      "5. 비중 부족 종목 매수\n"
                      "6. 비중 차이 2% 미만이면 유보\n"
                      "7. 거래 완료 후 실제 비중 재확인")


# ══════════════════════════════════════════
# Main App
# ══════════════════════════════════════════
class PortfolioApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.accent_palette = "Red"
        self.theme_cls.theme_style = "Light"
        self.title = "포트폴리오 최적화"
        return Builder.load_string(KV)

    def toggle_nav(self):
        nav = self.root.ids.nav_drawer
        if nav.state == "open":
            nav.set_state("close")
        else:
            nav.set_state("open")

    def switch_screen(self, name):
        self.root.ids.screen_manager.current = name
        self.root.ids.nav_drawer.set_state("close")


if __name__ == '__main__':
    PortfolioApp().run()
