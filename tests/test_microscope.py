from pyseq2500.microscope import Microscope


# from pyseq2500.utils import DEFAULT_CONFIG
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path


# @pytest_asyncio.fixture(
#     params=[
#         pytest.param("MockMicroscope", marks=pytest.mark.mock),
#         pytest.param("Microscope", marks=pytest.mark.hardware),
#     ],
#     scope="class",
# )
# async def microscope(request, fpga):
#     name = request.param
#     # Construct Microscope
#     m = Microscope()
#     if name == "MockMicroscope":
#         fcom = EmulatedFPGA(address="FPGACOM")
#         xcom = EmulatedXStage(name="XStage", address="XStageCOM")
#         ycom = EmulatedYStage(name="YStage", address="YStageCOM")
#         zcom = EmulatedZStage(name="ZStage", address="ZStageCOM")
#         tcom = EmulatedTiltMotor(name="TiltStage", address="FPGA")
#         glcom = EmulatedLaser(name="GreenLaser", address="LaserCOM")
#         rlcom = EmulatedLaser(name="RedLaser", address="LaserCOM")
#         gfwcom = EmulatedOptics(name="GreenFilterWheel", address="FPGA")
#         rfwcom = EmulatedOptics(name="RedFilterWheel", address="FPGA")
#         ecom = EmulatedOptics(name="EmissionFilter", address="FPGA")
#         scom = EmulatedOptics(name="Shutter", address="FPGA")
#         dcam = dcamCOM(emulated=True)

#         instruments = {
#             "FPGA": FPGA(com=fcom),
#             "XStage": XStage(name="XStage", com=xcom),
#             "YStage": YStage(name="YStage", com=ycom),
#             "ZStage": ZStage(name="ZStage", com=zcom),
#             "TiltStage": TiltStage(name="TiltStage", com=tcom),
#             "Lasers": {
#                 "green": Laser(name="GreenLaser", com=glcom, color="green"),
#                 "red": Laser(name="RedLaser", com=rlcom, color="red"),
#             },
#             "FilterWheels": {
#                 "green": FilterWheel(name="GreenFilterWheel", com=gfwcom),
#                 "red": FilterWheel(name="RedFilterWheel", com=rfwcom),
#             },
#             "EmissionFilter": EmissionFilter(name="EmissionFilter", com=ecom),
#             "Shutter": Shutter(name="Shutter", com=scom),
#             "Camera": TDICameras(com=dcam),
#         }

#         # Emulated coms for 3 motors
#         coms = {}
#         for i in range(1, 4):
#             coms[i] = EmulatedTiltMotor(name=f"TiltMotor{i}", address="FPGA")
#             await coms[i].connect()
#         # Assign emulated coms to fake stage motors
#         for id, com in coms.items():
#             instruments["TiltStage"].tilts[id] = TiltMotor(
#                 name=f"TiltMotor{id}", com=com
#             )

#         m.instruments = instruments
#         m.name = "MockMicroscope"
#     else:
#         # Attach fpga com fixture
#         fpga_instruments = [
#             "FPGA",
#             "ZStage",
#             "TiltStage",
#             "EmissionFilter",
#             "Shutter",
#         ]
#         for fi in fpga_instruments:
#             m.instruments[fi].com = fpga
#         for color in ["green", "red"]:
#             m.instruments["FilterWheels"][color].com = fpga

#     # Start async worker
#     m.start()

#     await m._connect()
#     await m._configure({})

#     # Do tests
#     yield m

#     # Shutdown instruments
#     await m._shutdown()

#     # Shutdown async worker gracefully
#     try:
#         m._worker_task.cancel()
#     except asyncio.CancelledError:
#         await m._worker_task

# async def test_px_to_step(self, microscope: Microscope, fc_A_roi):
#     """Test pixel to step conversion."""

#     x_step, y_step = microscope.px_to_step(100, 200, fc_A_roi.stage.model_dump())

#     assert isinstance(x_step, int)
#     assert isinstance(y_step, int)


@pytest.mark.microscope
@pytest.mark.asyncio
class TestMicroscope:
    @pytest_asyncio.fixture(autouse=True)
    async def test_init(self, microscope: Microscope):
        await microscope._initialize()
        # await microscope._configure({})

        all_status = []
        for instrument in microscope.iter_instruments:
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

    async def test_px_to_step(self, microscope: Microscope, fc_A_roi):
        """Test pixel to step conversion."""

        x_step, y_step = microscope.px_to_step(100, 200, fc_A_roi)

        assert isinstance(x_step, int)
        assert isinstance(y_step, int)

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

    async def test_scan(self, microscope: Microscope, fc_A_roi, fc_B_roi, mocker):
        if microscope.name == "MockMicroscope":
            # Use fc_A_roi for mock because it is smaller
            x = fc_A_roi.stage.x_init
            z = fc_A_roi.stage.z_init
            name = fc_A_roi.name
            image_dir = Path(fc_A_roi.image.image_dir)
            mock_save_image = mocker.patch.object(microscope.Camera, "save_image")
            await microscope._scan(fc_A_roi)
            assert mock_save_image.call_count == fc_A_roi.stage.nx
            for n in range(fc_A_roi.stage.nx):
                x_pos = n * fc_A_roi.stage.x_step + x
                im_name = f"s{name}_x{x_pos}_z{z}"
                assert mock_save_image.call_args_list[n].args[0] == im_name
                assert mock_save_image.call_args_list[n].args[1] == image_dir
        else:
            # Use fc_B_roi for actual sequence, because it is large enough that it won't throw a TDI error
            x = fc_B_roi.stage.x_init
            z = fc_B_roi.stage.z_init
            name = fc_B_roi.name
            image_dir = Path(fc_B_roi.image.image_dir)
            await microscope._scan(fc_B_roi)
            for n in range(fc_B_roi.stage.nx):
                x_pos = n * fc_B_roi.stage.x_step + x
                im_name = f"s{name}_x{x_pos}_z{z}"
                written_files = list(image_dir.glob(f"*_{im_name}.tiff"))
                assert len(written_files) == 4

    async def test_focus_stack(self, microscope: Microscope):
        focus_stack = await microscope._focus_stack(2000, 62000, 200)
        assert focus_stack.shape == (4, 200, 16, 2048)

    async def test_find_focus(self, microscope: Microscope, fc_A_roi):
        assert await microscope._find_focus(fc_A_roi)

    async def test_tilt_stack(
        self,
        microscope: Microscope,
        fc_A_roi,
        fc_B_roi,
        mocker,
        tilt_init=21500,
        tilt_last=22000,
        tilt_step=250,
    ):
        if microscope.name == "MockMicroscope":
            x = fc_A_roi.stage.x_init
            image_dir = Path(fc_A_roi.image.image_dir)
            name = fc_A_roi.name
            mock_save_image = mocker.patch.object(microscope.Camera, "save_image")
            await microscope._tilt_stack(
                fc_A_roi, tilt_init=tilt_init, tilt_last=tilt_last, tilt_step=tilt_step
            )
            c = 0
            for n in range(fc_A_roi.stage.nx):
                x_pos = n * fc_A_roi.stage.x_step + x
                for t in range(tilt_init, tilt_last, tilt_step):
                    im_name = f"s{name}_x{x_pos}_z{t}"
                    assert mock_save_image.call_args_list[c].args[0] == im_name
                    assert mock_save_image.call_args_list[c].args[1] == image_dir
                    c += 1
        else:
            x = fc_B_roi.stage.x_init
            image_dir = Path(fc_B_roi.image.image_dir)
            name = fc_B_roi.name
            await microscope._tilt_stack(
                fc_B_roi, tilt_init=tilt_init, tilt_last=tilt_last, tilt_step=tilt_step
            )
            for n in range(fc_B_roi.stage.nx):
                x_pos = n * fc_B_roi.stage.x_step + x
                for n, t in enumerate(range(tilt_init, tilt_last, tilt_step)):
                    im_name = f"s{name}_x{x_pos}_z{t}"
                    print(image_dir)
                    print(im_name)
                    written_files = list(image_dir.glob(f"*_{im_name}.tiff"))
                    assert len(written_files) == 4
