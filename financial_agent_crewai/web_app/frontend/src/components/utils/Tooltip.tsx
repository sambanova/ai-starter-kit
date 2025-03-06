interface TooltipProps {
  children: React.ReactNode;
  text: string;
}

const Tooltip = ({ children, text }: TooltipProps) => {
  return (
    <div className="relative group">
      {children}
      <div className="sn-tooltip sn-border absolute left-1/2 transform -translate-x-1/2 bottom-full mb-2 hidden group-hover:block text-md rounded py-1 px-2 z-10 w-xs whitespace-normal break-words text-center">
        {text}
      </div>
    </div>
  );
};

export default Tooltip;
