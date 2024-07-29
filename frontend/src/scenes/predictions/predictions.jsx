import { Canvas } from "@react-three/fiber";
import { useState, useRef } from "react";
import Select from "react-select";
import Model from "/public/Model";
import DashboardBox from "@/components/DashboardBox";
import { useTheme } from "@mui/material";
import {
  Box,
  Button,
  Alert,
  AlertTitle,
  IconButton,
  Collapse,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import demo from "/public/IMG_2850.mov";
import FlexBetween from "@/components/FlexBetween";

// import Alert from "@mui/material/Alert";
// import AlertTitle from "@mui/material/AlertTitle";

function Predictions() {
  const palette = useTheme();
  console.log(palette);

  const gridTemplateLargeScreens = `
    "e e e"
    "a b c"
    "d d d"
`;

  const tempData = 0;
  const [angleData, setAngleData] = useState([]);
  const [timeData, setTimeData] = useState([]);

  const [open, setOpen] = useState(true);

  const isPlaying = useRef(true);
  const setPlay = () => {
    isPlaying.current = !isPlaying.current;
    console.log(isPlaying.current);
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
      jsonFileDownload();
    }
  };

  const options = [
    { value: 1, label: "1 Sensor Model" },
    { value: 6, label: "6 Sensor Model" },
  ];

  const [selectedModel, setSelectedModel] = useState(1);
  console.log(selectedModel);

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

  const jsonFileDownload = () => {
    console.log(angleData);
    console.log(timeData);

    //const tmpArray = (arr, n) => arr.map(x => x[n]);

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

  return (
    <>
      <Collapse in={open}>
        <Alert
          action={
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={() => {
                setOpen(false);
              }}
            >
              <CloseIcon fontSize="inherit" />
            </IconButton>
          }
          sx={{ mb: 2 }}
        >
          {tempData} seconds of input data loaded to model.
        </Alert>
      </Collapse>
      <div>
        <Box
          width="100%"
          height="100%"
          display="grid"
          gap="1.5rem"
          sx={{
            // modify the minmax for the size of the graphs
            // fr -> fractional units (split each box evenly)
            gridTemplateColumns: "repeat(3, minmax(370px, 1fr))",
            gridTemplateRows: "repeat(1, minmax(90px, 1fr))",
            gridTemplateAreas: gridTemplateLargeScreens,
          }}
        >
          <Box gridArea="e">
            <Select 
              defaultValue={options[0]}
              onChange={setSelectedModel}
              options={options}
            />
          </Box>
          <DashboardBox gridArea="a">
            <Canvas id="modelCanvas">
              <Model
                passAngleData={setAngleData}
                passTimeData={setTimeData}
                selectedModel={selectedModel}
                isPlaying={isPlaying}
                currentFrame={currentFrame}
                rotDir={rotDir}
              />
            </Canvas>
          </DashboardBox>
          <DashboardBox gridArea="b">
            <video width="300" height="400" loop={true} autoPlay controls>
              <source src={demo} type="video/mp4" />
            </video>
          </DashboardBox>
          <DashboardBox gridArea="c">
            <video width="300" height="400" loop={true} autoPlay controls>
              <source src={demo} type="video/mp4" />
            </video>
          </DashboardBox>
          <DashboardBox
            gridArea="d"
            sx={{
              display: "flex",
              justifyContent: "space-evenly",
            }}
          >
            <Button variant="contained" type="button" onClick={setRotLeft}>
              Rotate Left
            </Button>
            <Button variant="contained" type="button" onClick={setPlay}>
              Pause/Play
            </Button>
            <Button variant="contained" type="button" onClick={setRotRight}>
              Rotate Right
            </Button>
            <Button variant="contained" type="button">
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
          </DashboardBox>
        </Box>
      </div>
    </>
  );
}

export default Predictions;
