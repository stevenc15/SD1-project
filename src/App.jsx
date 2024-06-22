import { useState, Suspense } from 'react'
import './App.css'

import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import Arm from '../public/Arm'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Canvas>
        <ambientLight />
        <OrbitControls />
        <Suspense fallback={null}>
          <Arm />
        </Suspense>
      </Canvas>
    </>
  )
}

export default App
