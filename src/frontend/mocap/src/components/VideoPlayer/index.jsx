// src/components/VideoPlayer.js

import { BigPlayButton, ControlBar,
        LoadingSpinner, Player, PlayToggle } from "video-react"
import "video-react/dist/video-react.css"
import { useEffect, useState } from "react"

export function VideoPlayer({
    src,
    onPlayerChange = () => { },
    onChange = () => { },
    startTime = undefined,
}) {
    const [player, setPlayer] = useState(undefined)
    const [playerState, setPlayerState] = useState(undefined)

    useEffect(() => {
        if (playerState) {
            onChange(playerState)
        }
    }, [playerState])

    useEffect(() => {
        onPlayerChange(player)

        if (player) {
            player.subscribeToStateChange(setPlayerState)
        }
    }, [player])

      const handleDownload = () => {
        if (player.current) {
            const videoSrc = player.current.src;

            if (videoSrc) {
                const link = document.createElement('a');
                link.href = videoSrc;
                link.download = 'video.mp4';  // or use a more dynamic filename
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                alert('No video available to download');
            }
        }
    };

    return (
        <div className={"video-player"}>
            <Player
                ref={(player) => {
                    setPlayer(player)
                }}
                startTime={startTime}
            >
                <source src={src} />
                <BigPlayButton position="center" />
                <LoadingSpinner />
                <ControlBar autoHide={false} disableDefaultControls={true}>
                    <PlayToggle />
                </ControlBar>
            </Player>
        </div>
    )
}
