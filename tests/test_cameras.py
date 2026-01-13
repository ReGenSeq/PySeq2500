from pyseq2500.camera import TDICameras
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockCamera", marks=pytest.mark.mock),
        pytest.param("Camera", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def cameras(request):
    cameras = TDICameras()
    await cameras.configure()

    yield cameras

    await cameras.shutdown()


@pytest.mark.camera
@pytest.mark.asyncio
class TestCameras:
    @pytest.mark.diagnostic
    async def test_init(self, cameras: TDICameras):
        await cameras.initialize()
        for cam in cameras:
            assert cam.sensor_mode == "TDI"

    @pytest.mark.diagnostic
    async def test_area(self, cameras: TDICameras):
        await cameras.setAREA()
        for cam in cameras:
            assert cam.sensor_mode == "AREA"

    async def test_shutdown(self, cameras: TDICameras):
        await cameras.shutdown()
        assert len(cameras.cams) == 0
