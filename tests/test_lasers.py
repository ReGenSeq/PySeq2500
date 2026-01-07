from pyseq2500.com import COM_DICT
from pyseq2500.laser import Laser, EmulatedLaser
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockLaser", marks=pytest.mark.mock),
        pytest.param("greenLaser", marks=pytest.mark.hardware),
        pytest.param("redLaser", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def laser(request) -> Laser:
    name = request.param

    if name == "MockLaser":
        com = EmulatedLaser(name="greenLaser", address="LaserCOM")
    else:
        com = COM_DICT[request.param]

    color = "red" if "red" in name else "green"
    laser = Laser(name=com.name, com=com, color=color)

    # Connect to COM
    await laser.com.connect()
    assert laser.connected

    yield laser

    await laser.shutdown()


@pytest.mark.optical
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestLaser:
    async def test_init(self, laser: Laser):
        await laser.initialize()

    async def test_power(self, laser: Laser):
        await laser.set_power(50)
        assert laser.power == 50

    async def test_status(self, laser: Laser):
        assert await laser.status()

    async def test_shutdown(self, laser: Laser):
        await laser.shutdown()
        assert not await laser.status()
