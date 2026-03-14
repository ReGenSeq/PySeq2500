from pyseq2500.com import COM_DICT
from pyseq2500.laser import Laser, EmulatedLaser
import pytest
import pytest_asyncio
import asyncio
from typing import List


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockLaser", marks=pytest.mark.mock),
        pytest.param("Laser", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def lasers(request) -> List[Laser]:
    lasers = []
    for c in ["Red", "Green"]:
        name = f"{c}Laser"
        if request.param == "MockLaser":
            com = EmulatedLaser(name=name, address="LaserCOM")
        else:
            com = COM_DICT[name]

        lasers.append(Laser(name=name, com=com, color=c.lower()))

    yield lasers

    # Connect to COM

    _ = []
    for laser in lasers:
        _.append(laser.shutdown())
    await asyncio.gather(*_)


@pytest.mark.optical
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestLaser:
    async def test_init(self, lasers: List[Laser]):
        _ = []
        for laser in lasers:
            _.append(laser.connect())
        await asyncio.gather(*_)

        _ = []
        for laser in lasers:
            assert laser.connected
            _.append(laser.initialize())
        await asyncio.gather(*_)

    async def test_power(self, lasers: List[Laser]):
        # Test is laser power gets within 5 %

        set_power = 50
        _ = []
        for laser in lasers:
            _.append(laser.set_power(set_power))
        await asyncio.gather(*_)

        powered_on = [False, False]
        try:
            async with asyncio.timeout(set_power):
                while all(powered_on):
                    for i, laser in enumerate(lasers):
                        act_power = await laser.get_power()
                        if 0.05 > abs(act_power / set_power - 1):
                            powered_on[i] = True
                            assert True
                        else:
                            await asyncio.sleep(0.5)
        except asyncio.TimeoutError:
            assert False

    async def test_status(self, lasers: List[Laser]):
        for laser in lasers:
            assert await laser.status()

    async def test_shutdown(self, lasers: List[Laser]):
        _ = []
        for laser in lasers:
            _.append(laser.shutdown())
        await asyncio.gather(*_)

        for laser in lasers:
            assert not await laser.status()
