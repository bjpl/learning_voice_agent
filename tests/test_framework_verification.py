"""
Framework Verification Tests
Simple tests to verify pytest setup works correctly
"""
import pytest


class TestPytestFramework:
    """Verify pytest is configured correctly"""

    def test_basic_assertion(self):
        """Test basic assertions work"""
        assert True
        assert 1 + 1 == 2
        assert "hello" in "hello world"

    def test_list_operations(self):
        """Test list operations"""
        test_list = [1, 2, 3]
        assert len(test_list) == 3
        assert 2 in test_list

    def test_dict_operations(self):
        """Test dictionary operations"""
        test_dict = {"key": "value"}
        assert "key" in test_dict
        assert test_dict["key"] == "value"

    @pytest.mark.asyncio
    async def test_async_support(self):
        """Test async/await support"""
        async def async_function():
            return "success"

        result = await async_function()
        assert result == "success"


class TestFixtures:
    """Test fixture functionality"""

    @pytest.fixture
    def sample_data(self):
        """Sample fixture"""
        return {"test": "data"}

    def test_fixture_works(self, sample_data):
        """Test fixtures are working"""
        assert sample_data["test"] == "data"


class TestMarkers:
    """Test pytest markers"""

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test unit marker"""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test integration marker"""
        assert True
