import DashboardBox from '@/components/DashboardBox'
import React from 'react'

type Props = {}

const Row4 = (props: Props) => {
  return (
    <>
        <DashboardBox gridArea="m"> </DashboardBox>
        <DashboardBox gridArea="n"> </DashboardBox>
        <DashboardBox gridArea="o"> </DashboardBox>
    </>
  )
}

export default Row4