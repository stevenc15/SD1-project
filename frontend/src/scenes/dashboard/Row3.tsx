import DashboardBox from '@/components/DashboardBox'
import React from 'react'

type Props = {}

function Row3(props: Props) {
  return (
    <>
      <DashboardBox gridArea="i"> </DashboardBox>
      <DashboardBox gridArea="j"> </DashboardBox>
      <DashboardBox gridArea="k"> </DashboardBox>
    </>
  )
}

export default Row3