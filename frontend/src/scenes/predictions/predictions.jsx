import { Canvas } from "@react-three/fiber";
import { useState, useRef, useEffect } from "react";
import Select from "react-select";
import Model from "/public/Model";
import DashboardBox from "@/components/DashboardBox";
import { useTheme } from "@mui/material";
import {
  Box,
  Button,
  Alert,
  IconButton,
  Collapse,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import LiveSwap from "../dashboard/LiveSwap";


// import Alert from "@mui/material/Alert";
// import AlertTitle from "@mui/material/AlertTitle";
// import demo from "/public/IMG_2850.mov";
// import FlexBetween from "@/components/FlexBetween";
import LiveRow from "../dashboard/LiveRow";

function Predictions() {
  const palette = useTheme();

  const gridTemplateLargeScreens = `
    "a b . "
`;

const gridTemplateButtons = `
    "d"
`;

  const [angleData, setAngleData] = useState([]);
  const [timeData, setTimeData] = useState([]);
  const [elapsedTime, setElapsedTime] = useState()

  const [open, setOpen] = useState(true)
  // const open = use(true);
  // const setOpen = () => {
  //   open.current = true
  // }
  // const setClose = () => {
  //   open.current = false
  // };

  const isPlaying = useRef(true);
  const setPlay = () => {
    isPlaying.current = !isPlaying.current;
    console.log("setPlay: " + isPlaying.current)
  };

  const currentFrame = useRef(0);
  const recStatus = useRef(0);
  const frame1 = useRef(0);
  const frame2 = useRef(0);

  const setRecStatus = () => {
    console.log(recStatus.current);
    if (recStatus.current == 0) {
      frame1.current = currentFrame.current;
      recStatus.current = recStatus.current + 1;
    }
    if (recStatus.current == 1) {
      frame2.current = currentFrame.current;
      recStatus.current = 0;
      console.log(frame1.current + " " + frame2.current);
      jsonFileDownloadRec();
    }
  };

  const options = [
    { value: 1, label: "1 Sensor Model" },
    { value: 6, label: "6 Sensor Model" },
  ];

  const [selectedModel, setSelectedModel] = useState(1);
  const setModelVersion = () => {
    if(selectedModel == 1)
    {
      setSelectedModel(6)
    }
    else if(selectedModel == 6)
    {
      setSelectedModel(1)
    }
    console.log("Selected model changed to: " + selectedModel)
  }

  const rotDir = useRef(0);
  const setRotLeft = () => {
    if (rotDir.current == -1) {
      rotDir.current = 0;
    } else {
      rotDir.current = -1;
    }
  };
  const setRotRight = () => {
    if (rotDir.current == 1) {
      rotDir.current = 0;
    } else {
      rotDir.current = 1;
    }
  };

  const resetState = () => {
    currentFrame.current = 0
    console.log("Reset.")
  }

  const jsonFileDownload = () => {
    let tmpArray = angleData;
    let json_data = JSON.stringify(tmpArray);
    const fileName = "joint_data.json";
    const data = new Blob([JSON.stringify(json_data)], { type: "text/json" });
    const jsonURL = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    document.body.appendChild(link);
    link.href = jsonURL;
    link.setAttribute("download", fileName);
    link.click();
    document.body.removeChild(link);
  };

  const jsonFileDownloadRec = () => {
    if(frame1 > frame2)
    {
      let tmpFrame = frame1.current
      frame1.current = frame2.current
      frame2.current = tmpFrame
    }
    else if (frame1 == frame2)
    {
      jsonFileDownload();
    }
    let tmpArray = angleData.slice(frame1, frame2)
    let json_data = JSON.stringify(tmpArray);
    const fileName = "joint_data.json";
    const data = new Blob([JSON.stringify(json_data)], { type: "text/json" });
    const jsonURL = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    document.body.appendChild(link);
    link.href = jsonURL;
    link.setAttribute("download", fileName);
    link.click();
    document.body.removeChild(link);
  };

  useEffect(() => {
    console.log("MODEL CHANGED: " + selectedModel)
    setOpen(true)
    setTimeout(() => {
      // setClose();
      setOpen(false);
    }, 3000);
  }, [selectedModel]);

  return (
    <>
      <Collapse in={open}>
        <Alert
          action={
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={ ()=>{setOpen(false)} }
            >
              <CloseIcon fontSize="inherit" />
            </IconButton>
          }
          sx={{ mb: 2 }}
        >
          Model {selectedModel} loaded.
        </Alert>
      </Collapse>
        <Select 
              defaultValue={options[0]}
              onChange={setModelVersion}
              options={options}
            />
        <Box
          width="100%"
          height="100%"
          display="grid"
          gap="6.0rem"
          sx={{
            gridTemplateColumns: "repeat(2, minmax(10px, 2fr))",
            gridTemplateRows: "repeat(1, minmax(100px, 2fr))",
            gridTemplateAreas: gridTemplateLargeScreens,
          }}
        >
          <DashboardBox height={600} width={670}my={4} sx={{backgroundColor: palette.palette.grey[900]}}>
          <Canvas id="modelCanvas">
              <Model
                passAngleData={setAngleData}
                passTimeData={setTimeData}
                passElapsed={setElapsedTime}
                selectedModel={selectedModel}
                isPlaying={isPlaying}
                currentFrame={currentFrame}
                rotDir={rotDir}
              />
            </Canvas>
          </DashboardBox>
          {/* <DashboardBox height={400} width={470} my={4}> 
            <video width="500" height="400" loop={true} autoPlay controls>
              <source src={demo} type="video/mp4" />
            </video>
          </DashboardBox> */}
          <DashboardBox height={600} width={670} my={4} gridArea={'b'}>
            {/* <LiveRow></LiveRow> */}
            <LiveSwap modelVersion={selectedModel}></LiveSwap>
          </DashboardBox>
        </Box>
        <Box display="grid"
          gap="2rem" sx={{
            gridTemplateColumns: "repeat(6, minmax(5px, 1fr))",
            gridTemplateRows: "repeat(1, minmax(1px, 1fr))",
            gridTemplateAreas: gridTemplateButtons,
          }}>
          <Button variant="contained" type="button" onClick={setRotLeft}>
            Rotate Left
          </Button>
          <Button variant="contained" type="button" onClick={setPlay}>
            Pause/Play
          </Button>
          <Button variant="contained" type="button" onClick={setRotRight}>
            Rotate Right
          </Button>
          <Button variant="contained" type="button" onClick={resetState}>
            Reset Camera
          </Button>
          <Button variant="contained" type="button" onClick={setRecStatus}>
            Start/Stop/Download Rec
          </Button>
          <Button
            variant="contained"
            type="button"
            onClick={jsonFileDownload}
          >
            Download Full Recording
          </Button>
        </Box>
    </>
  );
}

export default Predictions;
