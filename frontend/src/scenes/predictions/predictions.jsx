import { Canvas } from '@react-three/fiber'
import { useState, useRef } from 'react'
import Select from 'react-select'
import Model from '/public/Model'
import { Box } from '@mui/material'
import demo from '/public/IMG_2850.mov'

function Predictions() {

  const [angleData, setAngleData] = useState([]);
  const [timeData, setTimeData] = useState([]);

  const isPlaying = useRef(true)
  const setPlay = () => {
    isPlaying.current = ! isPlaying.current
    console.log (isPlaying.current)
  }
  
  const currentFrame = useRef(0)
  const setFrame = () => {
    console.log(currentFrame.current)
  }

  setFrame();

  const recStatus = useRef(0)
  const setRecStatus = () => {
    recStatus.current = recStatus.current + 1
    if (recStatus.current > 1){
      recStatus.current = 0
      jsonFileDownload();
    }
  }

  // const [recStatus, setRecStatus] = useState("0")
  // const [recStatusMessage, setRecStatMsg] = useState("Start Recording")

  const options = [
    { value: 1, label: '1 Sensor Model' },
    { value: 6, label: '6 Sensor Model' }
  ]

  const [selectedModel, setSelectedModel] = useState(1);
  console.log(selectedModel)

  const jsonFileDownload = () => {

    console.log(angleData)
    console.log(timeData)

    //const tmpArray = (arr, n) => arr.map(x => x[n]);

    let tmpArray = angleData
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
      <Select
        defaultValue={options[0]}
        onChange={setSelectedModel}
        options={options}
      />
      <Box
        height={500}
        width={1000}
        my={4}
        display="flex"
        alignItems="center"
        gap={4}
        p={2}
        sx={{ border: '2px solid grey' }}>
        <Canvas>
          <Model passAngleData={setAngleData} passTimeData={setTimeData} selectedModel={selectedModel} isPlaying={isPlaying} currentFrame={currentFrame}/>
        </Canvas>
        <video width="500" height="500" loop={true} autoPlay controls >
          <source src={demo} type="video/mp4" />
        </video>
      </Box>
      <button type="button">Rotate Left</button>
      <button type="button">Reset Camera</button>
      <button type="button">Rotate Right</button>
      {/* <button type="button" onClick={updateRecStatus}>{recStatusMessage}</button> */}
      <button type="button" onClick={setRecStatus}>Start/Stop/Download Rec</button>
      <button type="button" onClick={jsonFileDownload}>Download Full Recording</button>
      <button type="button" onClick={setPlay}>Pause/Play</button>
    </>
  )
}

export default Predictions
