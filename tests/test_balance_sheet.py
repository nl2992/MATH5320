"""
tests/test_balance_sheet.py
Unit tests for balance-sheet helpers in src/risk/regulatory.py.
"""
import pytest
from src.risk.regulatory import (
    balance_sheet_equity,
    balance_sheet_after_asset_loss,
    leverage_ratio,
)


class TestBalanceSheetEquity:
    def test_basic(self):
        assert balance_sheet_equity(1000.0, 800.0) == pytest.approx(200.0)

    def test_zero_liabilities(self):
        assert balance_sheet_equity(500.0, 0.0) == pytest.approx(500.0)

    def test_zero_assets(self):
        assert balance_sheet_equity(0.0, 0.0) == pytest.approx(0.0)

    def test_insolvent_entity_returns_negative(self):
        """Equity can be negative (insolvency)."""
        assert balance_sheet_equity(100.0, 150.0) == pytest.approx(-50.0)

    def test_negative_assets_raises(self):
        with pytest.raises(ValueError, match="assets"):
            balance_sheet_equity(-1.0, 0.0)

    def test_negative_liabilities_raises(self):
        with pytest.raises(ValueError, match="liabilities"):
            balance_sheet_equity(100.0, -1.0)


class TestBalanceSheetAfterAssetLoss:
    def test_solvent_after_loss(self):
        result = balance_sheet_after_asset_loss(1000.0, 600.0, 100.0)
        assert result["assets_post"] == pytest.approx(900.0)
        assert result["liabilities"] == pytest.approx(600.0)
        assert result["equity_post"] == pytest.approx(300.0)
        assert result["solvent"] is True

    def test_exactly_insolvent(self):
        # equity_post = 0  →  still solvent (boundary)
        result = balance_sheet_after_asset_loss(1000.0, 800.0, 200.0)
        assert result["equity_post"] == pytest.approx(0.0)
        assert result["solvent"] is True  # equity_post >= 0

    def test_insolvent_after_large_loss(self):
        result = balance_sheet_after_asset_loss(1000.0, 800.0, 300.0)
        assert result["equity_post"] == pytest.approx(-100.0)
        assert result["solvent"] is False

    def test_zero_loss(self):
        result = balance_sheet_after_asset_loss(1000.0, 400.0, 0.0)
        assert result["assets_post"] == pytest.approx(1000.0)
        assert result["equity_post"] == pytest.approx(600.0)
        assert result["solvent"] is True

    def test_liabilities_unchanged(self):
        result = balance_sheet_after_asset_loss(500.0, 300.0, 50.0)
        assert result["liabilities"] == pytest.approx(300.0)

    def test_negative_assets_raises(self):
        with pytest.raises(ValueError, match="assets"):
            balance_sheet_after_asset_loss(-1.0, 0.0, 0.0)

    def test_negative_liabilities_raises(self):
        with pytest.raises(ValueError, match="liabilities"):
            balance_sheet_after_asset_loss(100.0, -1.0, 0.0)

    def test_negative_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            balance_sheet_after_asset_loss(100.0, 0.0, -10.0)


class TestLeverageRatio:
    def test_typical(self):
        # equity 200, assets 1000  →  0.20
        assert leverage_ratio(200.0, 1000.0) == pytest.approx(0.20)

    def test_fully_equity_financed(self):
        assert leverage_ratio(500.0, 500.0) == pytest.approx(1.0)

    def test_small_equity(self):
        assert leverage_ratio(8.0, 100.0) == pytest.approx(0.08)

    def test_zero_assets_raises(self):
        with pytest.raises(ValueError, match="positive"):
            leverage_ratio(10.0, 0.0)

    def test_negative_assets_raises(self):
        with pytest.raises(ValueError, match="positive"):
            leverage_ratio(10.0, -1.0)

    def test_negative_equity_allowed(self):
        """Insolvent entity: negative equity / assets is meaningful."""
        assert leverage_ratio(-50.0, 500.0) == pytest.approx(-0.10)
