from pyseq2500.microscope import Microscope
from pyseq2500.xstage import EmulatedXStage, XStage
from pyseq2500.ystage import EmulatedYStage, YStage
from pyseq2500.zstage import EmulatedZStage, ZStage
from pyseq2500.tiltstage import EmulatedTiltMotor, TiltStage, TiltMotor
from pyseq2500.fpga import EmulatedFPGA, FPGA
from pyseq2500.laser import EmulatedLaser, Laser
from pyseq2500.optics import FilterWheel, EmissionFilter, Shutter, EmulatedOptics
from pyseq2500.camera import TDICameras, dcamCOM

# from pyseq2500.utils import DEFAULT_CONFIG
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from unittest.mock import patch


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockMicroscope", marks=pytest.mark.mock),
        pytest.param("Microscope", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def microscope(request, fpga):
    name = request.param
    # Construct Microscope
    m = Microscope(name=name)
    if name == "MockMicroscope":
        fcom = EmulatedFPGA(address="FPGACOM")
        xcom = EmulatedXStage(name="XStage", address="XStageCOM")
        ycom = EmulatedYStage(name="YStage", address="YStageCOM")
        zcom = EmulatedZStage(name="ZStage", address="ZStageCOM")
        tcom = EmulatedTiltMotor(name="TiltStage", address="FPGA")
        glcom = EmulatedLaser(name="GreenLaser", address="LaserCOM")
        rlcom = EmulatedLaser(name="RedLaser", address="LaserCOM")
        gfwcom = EmulatedOptics(name="GreenFilterWheel", address="FPGA")
        rfwcom = EmulatedOptics(name="RedFilterWheel", address="FPGA")
        ecom = EmulatedOptics(name="EmissionFilter", address="FPGA")
        scom = EmulatedOptics(name="Shutter", address="FPGA")
        dcam = dcamCOM(emulated=True)

        instruments = {
            "FPGA": FPGA(com=fcom),
            "XStage": XStage(name="XStage", com=xcom),
            "YStage": YStage(name="YStage", com=ycom),
            "ZStage": ZStage(name="ZStage", com=zcom),
            "TiltStage": TiltStage(name="TiltStage", com=tcom),
            "GreenLaser": Laser(name="GreenLaser", com=glcom, color="green"),
            "RedLaser": Laser(name="RedLaser", com=rlcom, color="red"),
            "GreenFilterWheel": FilterWheel(name="GreenFilterWheel", com=gfwcom),
            "RedFilterWheel": FilterWheel(name="RedFilterWheel", com=rfwcom),
            "EmissionFilter": EmissionFilter(name="EmissionFilter", com=ecom),
            "Shutter": Shutter(name="Shutter", com=scom),
            "Camera": TDICameras(com=dcam),
        }

        # Emulated coms for 3 motors
        coms = {}
        for i in range(1, 4):
            coms[i] = EmulatedTiltMotor(name=f"TiltMotor{i}", address="FPGA")
            await coms[i].connect()
        # Assign emulated coms to fake stage motors
        for id, com in coms.items():
            instruments["TiltStage"].tilts[id] = TiltMotor(
                name=f"TiltMotor{id}", com=com
            )

        m.instruments = instruments
    else:
        # Attach fpga com fixture
        fpga_instruments = [
            "FPGA",
            "ZStage",
            "TiltStage",
            "GreenFilterWheel",
            "RedFilterWheel",
            "EmissionFilter",
            "Shutter",
        ]
        for fi in fpga_instruments:
            m.instruments[fi].com = fpga

    # Start async worker
    m.start()

    await m._connect()
    await m._configure({})

    # Do tests
    yield m

    # Shutdown instruments
    await m._shutdown()

    # Shutdown async worker gracefully
    try:
        m._worker_task.cancel()
    except asyncio.CancelledError:
        await m._worker_task


@pytest.mark.microscope
@pytest.mark.asyncio
class TestMicroscope:
    async def test_init(self, microscope: Microscope):
        await microscope._initialize()
        # await microscope._configure({})

        all_status = []
        for instrument in microscope.instruments.values():
            if "Laser" in instrument.name:
                # Lasers not used so just test if connected
                status = instrument.connected
            elif "Shutter" in instrument.name:
                # Shutter is closed so status should read False
                status = not await instrument.status()
            else:
                status = await instrument.status()
            if not status:
                print(f"{instrument.name} failed to start")
            all_status.append(status)
        assert all(all_status)

    async def test_move(self, microscope: Microscope, fc_A_roi):
        x = fc_A_roi.stage.x_init
        y = fc_A_roi.stage.y_init
        z = fc_A_roi.stage.z_init
        t1 = fc_A_roi.stage.tilt1
        t2 = fc_A_roi.stage.tilt2
        t3 = fc_A_roi.stage.tilt3

        await microscope._move(x, y, z, t1, t2, t3)

        status = await asyncio.gather(
            microscope.XStage.status(),
            microscope.YStage.status(),
            microscope.ZStage.status(),
            microscope.TiltStage.status(),
        )
        assert all(status)

    async def test_scan(self, microscope: Microscope, fc_A_roi):
        x = fc_A_roi.stage.x_init
        z = fc_A_roi.stage.z_init
        name = fc_A_roi.name
        im_name = f"s{name}_x{x}_z{z}"
        image_dir = Path(fc_A_roi.image.image_dir)

        if microscope.name == "MockMicroscope":
            with patch.object(microscope.Camera, "save_image") as mock_save_image:
                await microscope._scan(fc_A_roi)
                mock_save_image.assert_called_once()
                args, kwargs = mock_save_image.call_args
                assert args[0] == im_name
                assert args[1] == image_dir
        else:
            await microscope._scan(fc_A_roi)
            written_files = list(image_dir.glob(f"*_{im_name}.tiff"))
            assert len(written_files) == 4
