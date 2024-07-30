import LiveRowOne from "./LiveRowOne"
import LiveRow from "./LiveRow"

function LiveSwap(props)
{
    console.log("Line Swap " + props.modelVersion + " Type: " + typeof(props.modelVersion))
    if(props.modelVersion == 1)
    {
        return(<LiveRowOne></LiveRowOne>);
    }
    else if(props.modelVersion == 6)
    {
        return(<LiveRow></LiveRow>)
    }
}

export default LiveSwap